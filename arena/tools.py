"""Tool surface exposed to the agent harness, with seeded hostility.

FROZEN-adjacent scaffold: part of `arena/` (instructor-owned), not
`harness/` (student-owned) — students consume `Tools`, they don't edit
it. `Tools` wraps `Corpus` access plus a sandboxed calculator and a
`submit()` endpoint, and emits one `tool_call` trace event per call
(`name` + `ok` are the fields `Trace.validate` requires — see
`arena/trace.py`'s CONTRACT section).

Seeded hostility. With `flaky=True` (the default), a fraction of calls
fail. The decision of *whether* call number `idx` (0-based, counted
across all four methods on one `Tools` instance — `self.calls` is that
same counter) fails is a pure function of `(seed, idx)`: a fresh
`random.Random(seed * 1_000_003 + idx)` per call, never Python's
built-in `hash()` (which is salted per-process unless
`PYTHONHASHSEED` is fixed, and would break the "identical seed ->
identical run" guarantee this lab's leaderboard depends on). Because
the roll is keyed purely on the running call index, replaying the same
sequence of tool calls with the same seed reproduces the exact same
pass/fail sequence — including across tool types, not just within one.

**IMPORTANT for Task 4/5/6/7 authors — the `flaky`-TAGGED DOC IN
`arena/corpus.py` HAS ZERO MECHANICAL CONNECTION TO THIS.** Flakiness
here is purely `(seed, call index)`-driven, as described above.
Fetching the doc tagged `"flaky"` does not make THIS OR ANY call more
or less likely to fail than fetching any other doc, and fetching a
non-`"flaky"`-tagged doc is exactly as likely to hit a flaky outcome.
The corpus's `flaky` tag is thematic scaffolding only (a document that
narratively explains why a sensor feed drops out, so the trap reads as
motivated in-world) — it exists to satisfy the trap-class coverage
requirement, not to select which fetches this module perturbs. If a
`retry` reference implementation or the scorer ever assumes "fetching
the flaky doc is what triggers flakiness," that assumption is wrong —
retry's real, intended target is the `ok=False`/`flaky_mode` signal on
ANY tool call, uniformly.

Failure modes (`_MODES_BY_TOOL` restricts which apply per tool):
    timeout   -> ok=False, no content, an obvious "please retry" error.
    truncate  -> ok=True, content silently shortened mid-sentence with
                 a visible "[TRUNCATED: ...]" marker — the trap is
                 that the answer LOOKS successful but lost information.
    noise     -> ok=True, content replaced with an unrelated garbled
                 payload carrying a "[NOISE: ...]" marker — the trap
                 is that a careless caller trusts `ok=True` and moves
                 on instead of recognising the marker and retrying.
`calc` and `submit` only ever time out (there's no "content" to
truncate/corrupt meaningfully for a bare number or a submit ack).

Safety. `calc` never calls `eval`/`exec`. It parses the expression with
`ast.parse(..., mode="eval")` and walks the resulting tree itself,
allow-listing only numeric constants and the basic arithmetic
operators — any other node shape (a `Call` such as
`__import__("os")`, an `Attribute`, a `Name`, ...) is rejected before
it is ever evaluated, so no arbitrary code can run.

Two independent, defence-in-depth bounds against a CPU/memory DoS via
arithmetic (not just via `**`, which was the only operator guarded in
an earlier revision — see fix-round-1 finding 1 below):

  1. `_MAX_EXPRESSION_CHARS` rejects an over-long expression BEFORE
     `ast.parse` ever runs. This alone defeats the *"exponential
     doubling via duplicated subexpression text"* attack shape — e.g.
     repeated `(x)*(x)` nesting with no `**` anywhere. Because there is
     no way to name/reuse a subexpression in bare arithmetic, any
     structure that makes a VALUE grow exponentially with nesting
     depth must also make the EXPRESSION TEXT grow exponentially with
     nesting depth (each level embeds two literal copies of the
     previous level). A cheap `len(expression)` check therefore catches
     this class at the door, in O(1), without parsing a single byte of
     a multi-megabyte payload.
  2. `_check_size` runs after EVERY `BinOp`/`UnaryOp`/`Constant` is
     evaluated (not only after `**`), rejecting any intermediate value
     whose bit length exceeds `_MAX_INT_BITS`, or that is a non-finite
     float (`inf`/`nan` — see fix-round-1 finding 4: a calculator
     reporting `inf`/`nan` as a successful result is itself a bug, not
     just a DoS concern). Because the check fires at every node and
     raises immediately, a duplicated-subexpression attack is rejected
     the FIRST time any intermediate crosses the cap — which happens at
     a shallow, fixed recursion depth (bit length roughly doubles per
     nesting level for a squaring pattern), so the tree walk aborts
     long before it would need to visit anywhere near an exponential
     number of nodes. `**` additionally gets a PRE-check (before the
     power is computed, not after) because unlike +-*/, a single `**`
     can turn two modest, in-budget operands into an astronomically
     large result in one step — computing it first and checking after
     would be too late.

Aliasing. `submit()` never hands the trace a live reference to the
caller's `report` dict (see `arena/trace.py`'s aliasing warning) — it
serialises the report to a `str` via `json.dumps` first. A `str` is
immutable, so a caller mutating their `report` dict after calling
`submit()` cannot retroactively change what the trace recorded.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import random
from dataclasses import dataclass

from arena.corpus import Corpus
from arena.trace import Trace

# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Restricted arithmetic parser — never eval().
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# --- DoS guards -------------------------------------------------------
#
# fix-round-1 finding 1 (CRITICAL): an earlier revision only bounded
# `**`. A Mult chain built from nested, duplicated parens —
# `(((x)*(x))*((x)*(x)))*...` — has no `**` anywhere and sailed
# straight through, hanging the process on a multi-million-digit int.
# Two independent bounds now apply, neither `**`-specific:

# 1. Reject an over-long expression before it is ever parsed. Because
#    bare arithmetic has no way to name/reuse a subexpression, ANY
#    structure that doubles a VALUE at each nesting level must also
#    double the EXPRESSION TEXT at each level (two literal copies of
#    the previous level are embedded). A cheap O(1) length check
#    therefore catches every such duplication attack at the door, for
#    any operator, without spending a single parse cycle on a
#    multi-megabyte payload. 2000 chars is far more than any legitimate
#    lab calculation needs.
_MAX_EXPRESSION_CHARS = 2000

# 2. Bound every INTERMEDIATE value produced while walking the tree,
#    not just the result of `**`. Applied after every BinOp/UnaryOp and
#    to every literal constant. ~30,000 decimal digits is generous for
#    any real calculation, but keeps every individual arithmetic op
#    (even Python's optimized big-int multiply) at low-millisecond
#    cost, and — critically — the check fires and raises at the FIRST
#    node (any node, not just the outermost) whose value crosses the
#    cap. For a doubling/squaring structure, bit length roughly doubles
#    per nesting level, so that first violation happens at a small,
#    fixed recursion depth (~15-20 levels for this cap) — the recursive
#    walk aborts via the raised exception long before it would need to
#    visit anywhere near an exponential number of AST nodes.
_MAX_INT_BITS = 100_000

# `**` gets an ADDITIONAL pre-check (before computing, not after):
# unlike +-*/, a single `**` can turn two modest, in-budget operands
# into an astronomically large result in one step (e.g. a ~90,000-bit
# base raised to the 1000th power) — computing it first and checking
# the result after would already be too late.
_MAX_POW_EXPONENT = 1000


def _check_size(value) -> None:
    """Reject a value that is already too large, or non-finite
    (inf/nan — fix-round-1 finding 4: a calculator silently succeeding
    with `inf`/`nan` is a correctness bug, not just a DoS concern).
    Called after every constant/BinOp/UnaryOp evaluation.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > _MAX_INT_BITS:
            raise ValueError(f"intermediate value too large (limit: {_MAX_INT_BITS} bits)")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result is not a finite number (inf/nan)")


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            _check_size(node.value)
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POW_EXPONENT:
                raise ValueError(f"exponent too large (limit: {_MAX_POW_EXPONENT})")
            if isinstance(left, int) and left.bit_length() * max(abs(right), 1) > _MAX_INT_BITS:
                raise ValueError("result of ** would be too large")
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        _check_size(result)
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        result = _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
        _check_size(result)
        return result
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


def _safe_arithmetic(expression: str):
    # NOTE for scanners: no call to the `eval()` builtin appears anywhere
    # in this module. `mode="eval"` below is an `ast.parse` option meaning
    # "parse a single expression" (as opposed to a statement) — it only
    # produces a syntax tree, it does not execute anything. The tree is
    # then walked by our own `_safe_eval` (a plain recursive function,
    # unrelated to the builtin of the same name) which allow-lists numeric
    # constants and arithmetic operators and rejects everything else
    # (Call, Attribute, Name, ...) before any computation happens.
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ValueError(
            f"expression too long ({len(expression)} chars, limit: "
            f"{_MAX_EXPRESSION_CHARS}) — refusing to parse"
        )
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree)


def _format_number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Flakiness
# ---------------------------------------------------------------------------

_FLAKY_RATE = 0.15
_MODES_BY_TOOL = {
    "search": ("timeout", "truncate", "noise"),
    "fetch_doc": ("timeout", "truncate", "noise"),
    "calc": ("timeout",),
    "submit": ("timeout",),
}


def _apply_truncate(content: str, rng: random.Random) -> str:
    if not content:
        return "[TRUNCATED: connection dropped before any data was received]"
    frac = 0.2 + rng.random() * 0.3
    keep = max(1, int(len(content) * frac))
    return content[:keep] + " [TRUNCATED: connection dropped]"


def _apply_noise(rng: random.Random) -> str:
    junk = "".join(rng.choice("XQZ#$%&@!?0123456789") for _ in range(24))
    return f"[NOISE: response corrupted, integrity check failed — payload:{junk}]"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class Tools:
    def __init__(self, corpus: Corpus, trace: Trace, seed: int, flaky: bool = True) -> None:
        self._corpus = corpus
        self._trace = trace
        self._seed = seed
        self._flaky = flaky
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def _roll(self, tool: str):
        """Decide whether the CURRENT call (index `self._calls`) is a
        flaky failure, purely as a function of `(self._seed,
        self._calls)`. Returns `(mode_or_None, rng)`; `rng` keeps
        advancing the same deterministic stream so mode-specific
        randomised content (truncation length, noise payload) is also
        reproducible for this call.
        """
        if not self._flaky:
            return None, None
        idx = self._calls
        rng = random.Random(self._seed * 1_000_003 + idx)
        if rng.random() >= _FLAKY_RATE:
            return None, rng
        modes = _MODES_BY_TOOL[tool]
        return modes[rng.randrange(len(modes))], rng

    def _emit_tool_call(self, name: str, ok: bool, **extra) -> None:
        self._trace.emit("tool_call", name=name, ok=ok, **extra)

    @staticmethod
    def _trace_preview(text: str) -> str:
        """Cap a caller-controlled string BEFORE handing it to
        `Trace.emit`, so a pathologically large input (e.g. the
        multi-megabyte Mult-chain `calc` expression from fix-round-1
        finding 1 — already rejected fast by `_safe_arithmetic`'s own
        length cap, but still a `str` sitting in this call's local
        scope) can't force an O(size) `json.dumps` inside `Trace.emit`'s
        own budget-fitting on every such call. `Trace.emit`'s own
        truncation (see `arena/trace.py`) is a safety net for
        otherwise-reasonable content; it is not a substitute for
        capping a KNOWN-attacker-controlled field at the source, since
        even deciding to truncate costs `Trace.emit` one `json.dumps`
        of the full original value first.
        """
        limit = 500
        if len(text) <= limit:
            return text
        return text[:limit] + f"...[preview truncated, {len(text)} chars total]"

    def search(self, query: str, k: int = 5) -> ToolResult:
        mode, rng = self._roll("search")
        results = self._corpus.search(query, k=k)
        payload = [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "snippet": (d.body[:180] + "…") if len(d.body) > 180 else d.body,
            }
            for d in results
        ]
        content = json.dumps(payload, ensure_ascii=False)
        ok, error = True, None

        if mode == "timeout":
            ok, content, error = False, "", f"timeout: search({query!r}) exceeded time budget"
        elif mode == "truncate":
            content = _apply_truncate(content, rng)
        elif mode == "noise":
            content = _apply_noise(rng)

        self._emit_tool_call("search", ok, query=self._trace_preview(query), k=k, n_results=len(results), flaky_mode=mode)
        self._calls += 1
        return ToolResult(ok=ok, content=content, error=error)

    def fetch_doc(self, doc_id: str) -> ToolResult:
        mode, rng = self._roll("fetch_doc")
        doc = self._corpus.get(doc_id)

        if doc is None:
            ok, content, error = False, "", f"doc not found: {doc_id}"
        else:
            ok, content, error = True, doc.body, None
            if mode == "timeout":
                ok, content, error = False, "", f"timeout: fetch_doc({doc_id!r}) exceeded time budget"
            elif mode == "truncate":
                content = _apply_truncate(content, rng)
            elif mode == "noise":
                content = _apply_noise(rng)

        self._emit_tool_call("fetch_doc", ok, doc_id=self._trace_preview(doc_id), flaky_mode=mode)
        self._calls += 1
        return ToolResult(ok=ok, content=content, error=error)

    def calc(self, expression: str) -> ToolResult:
        mode, _rng = self._roll("calc")
        try:
            value = _safe_arithmetic(expression)
            ok, content, error = True, _format_number(value), None
        except Exception as exc:  # never eval(); any parse/eval failure -> ok=False
            ok, content, error = False, "", f"invalid expression: {exc}"

        if mode == "timeout":
            ok, content, error = False, "", f"timeout: calc({expression!r}) exceeded time budget"

        self._emit_tool_call("calc", ok, expression=self._trace_preview(expression), flaky_mode=mode)
        self._calls += 1
        return ToolResult(ok=ok, content=content, error=error)

    def submit(self, report: dict) -> ToolResult:
        mode, _rng = self._roll("submit")
        report_json: str | None = None

        if not isinstance(report, dict):
            ok, content, error = False, "", "report must be a dict"
        else:
            try:
                report_json = json.dumps(report, sort_keys=True, ensure_ascii=False)
            except TypeError as exc:
                ok, content, error = False, "", f"report is not JSON-serialisable: {exc}"
            else:
                ok, content, error = True, "submitted", None

        if ok and mode == "timeout":
            ok, content, error = False, "", "timeout: submit exceeded time budget"

        self._emit_tool_call(
            "submit", ok,
            report_json=(report_json if report_json is not None else ""),
            flaky_mode=mode,
        )
        self._calls += 1
        return ToolResult(ok=ok, content=content, error=error)
