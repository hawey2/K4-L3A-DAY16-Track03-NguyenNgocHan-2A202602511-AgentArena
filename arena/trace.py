"""Trace schema and conformance gate for Agent Arena.

FROZEN — students may read this file but must never edit it (the
protect-symlinks-style convention for this lab is: `arena/` is the
scaffold, `harness/` is student-owned).

A `Trace` records one run of an agent against a brief as a sequence of
JSON-Lines events. `Trace.validate` is the trace-conformance GATE used
by the scorer: it is a PASS/FAIL check, never a fourth scored
dimension. It exists to force students to route every model call, tool
call, and middleware layer through the harness — bypass the harness and
the trace stops conforming.

Design note: `emit` is cheap and forgiving — it checks the event name
against `EVENTS`, checks that reserved fields aren't clobbered, and
automatically reshapes the record (truncating values, shortening
over-long field names, dropping surplus fields) as needed to keep the
emitted line within `_MAX_EMITTED_LINE_CHARS` while PRESERVING every
field `validate` requires (see CONTRACT FOR TASK 2/3/4 AUTHORS below).
It never rejects a call for being "too big." `validate` is strict and
must give a specific, machine-greppable reason string on failure so
students (and later tasks' tooling) can act on it directly.

Threat model: `validate` is a `@staticmethod` that accepts arbitrary
text, not just text produced by this module's `Trace.emit`/`to_jsonl`.
Students are motivated to game the trace-conformance gate directly (by
hand-writing JSONL that never touches the harness), and `validate`
decides whether their entry is scored at all — so it is DESIGNED TO
NEVER RAISE, including under adversarial input such as a JSON "bomb"
(pathologically deep nesting, e.g. `"["*200000 + "]"*200000`, crafted
to blow the C JSON decoder's native stack rather than Python's
catchable `RecursionError` machinery). Two defenses work together:
(1) a per-line length ceiling (`_MAX_TRACE_LINE_CHARS`, see below)
rejects grossly oversized lines before they are ever handed to
`json.loads`, so the expensive/dangerous parse is never attempted; (2)
the `json.loads` call itself is wrapped in a bare `except Exception`
rather than the narrower `except json.JSONDecodeError` — deliberately,
because a hand-crafted line can make the decoder raise things other
than `JSONDecodeError` (observed: `RecursionError` from deep nesting),
and at this specific boundary the never-raise invariant outranks
exception-type precision. Any malformed, adversarial, or merely
unexpected input is a `(False, "<specific reason>")`, never an
uncaught exception, because an exception here could crash the scorer
mid-run for the whole batch, not just the offending entry.

`_MAX_TRACE_LINE_CHARS = 100_000` — chosen with margin on the
JSON-bomb side: empirically (this CPython 3.14 build, see fix-round-2
verification), deeply nested JSON only starts risking a native stack
overflow past ~150,000 levels of array nesting (~300,000 characters) or
~150,000 levels of object nesting (~900,000 characters); 100,000
characters leaves at least 2x margin below the smallest observed
unsafe boundary. The ceiling is a coarse, fast first line of defense,
not a substitute for defense (2) above — a bomb shape that recurses at
fewer characters per nesting level (a different JSON dialect/encoder, a
future CPython, a smaller-stack platform) would sail under any length
ceiling and still needs the broad `except Exception` to be caught
safely.

CONTRACT FOR TASK 2/3/4 AUTHORS (Corpus, Tools, harness layers) — read
this if your code calls `Trace.emit`.

THE GUARANTEE. If a call to `Trace.emit` returns (i.e. it did not raise
for an unknown event name or a reserved-key clash), then the JSONL line
it appends is one that `Trace.validate` accepts — assuming you passed
the fields validate requires for that event kind (`name` + `ok` for
`tool_call`; `prompt_tokens` + `completion_tokens` for `model_call`).
`emit` will never turn a well-formed call of yours into a gate failure.
Concretely, you do NOT need to bound any of the following before
calling `emit`:
  * how large a field VALUE is (a full fetched-document body, a huge
    tool result, a 2 MB string, a list of a million ints — all fine);
  * how long a field NAME is;
  * how MANY fields you attach to one event.
`emit` reshapes whatever it is given until the record fits its
write-side budget `_MAX_EMITTED_LINE_CHARS = 90_000` — an internal
budget 10,000 characters (10%) below validate's `_MAX_TRACE_LINE_CHARS`
ceiling.

WHAT MAY BE LOST when a record is over budget (in this order — cheapest
loss first, and nothing at all is touched if the record already fits):
  1. Field VALUES are shortened. Strings keep a prefix and gain a
     visible `"...[truncated N chars]"` suffix; non-string values
     (lists, dicts, ...) can't be meaningfully sliced, so an oversized
     one is replaced with a short `"...[truncated <type> value, ~N
     chars serialized]"` marker string. The remaining budget is shared
     between fields smallest-first, so a single huge field absorbs the
     truncation while your small fields survive intact.
  2. Over-long field NAMES (> `_MAX_FIELD_NAME_CHARS`) are shortened to
     a prefix plus `"...[truncated name]"`, with a `~2`, `~3`, ...
     suffix appended if that shortening would collide with another key.
  3. Fields beyond `_MAX_EXTRA_FIELDS` are DROPPED outright, and the
     number dropped is recorded as `fields_dropped`.
Any record that was reshaped at all also carries `fields_truncated:
true`, so a reader can tell trace loss from agent behaviour.

WHAT IS NEVER LOST: the reserved fields (`seq`, `event`, `run_id`,
`seed`) and the fields `validate` requires for the record's event kind
(`_REQUIRED_FIELDS_BY_EVENT` — the SAME declaration `validate` itself
reads, so the two can never drift apart). Those keys are always
present in the emitted record. Their values may still be shortened if
they are themselves enormous — presence, not byte-exactness, is what
the gate checks — but the keys survive even in the worst case, so
"`emit` succeeded but the trace fails the gate for a missing required
field" cannot happen.

**Practical implication: do not rely on the trace to carry the
complete, byte-exact content of a very large tool result.** A
realistic fetched-document body (tens of KB, comfortably under the
90,000-char budget once the small record overhead is accounted for)
passes through completely untouched; only content that actually
threatens the budget is shortened. But if a later component (e.g.
`citation_checker`, a grounding scorer) needs the full original text,
keep your own reference to it — e.g. a `doc_id` the corpus can look up
— rather than assuming the trace preserves everything you logged.

COST. `Trace.emit` is `O(record size)` per call (one `json.dumps` for
the size check), not the strict `O(1)` of earlier drafts of this
module, but that's cheap in absolute terms. The reshaping work above
only runs on the rare over-budget record.

KNOWN LIMITATION (accepted). Because `emit` calls `json.dumps` to size
the record, a field value that Python itself cannot serialize will
raise from inside `emit`: a non-JSON-serialisable object raises
`TypeError`, a deeply-nested NATIVE list/dict raises `RecursionError`,
and an int beyond CPython's int-to-str digit limit raises
`ValueError`. These are caller bugs that would equally have raised at
`to_jsonl()` time; they are outside the guarantee above.

DECISION — `seq` is int-only (not float). There is no use case for a
fractional sequence number, and restricting to `int` makes `validate`
simpler to reason about and easier to make crash-proof against
hand-written JSONL (no need to worry about float precision/NaN/Infinity
edge cases in ordering comparisons). `Trace.emit` only ever produces
int `seq` values, so this is free for any student using the scaffold
normally; it only excludes traces that were hand-tampered to use
non-int `seq`.
"""

from __future__ import annotations

import json

# The five event kinds a conforming trace may contain, in the order
# they were introduced in the Day 16 lecture deck's middleware model.
EVENTS = ("agent_start", "model_call", "tool_call", "layer", "agent_end")

# Fields Trace.emit stamps on every record automatically. A caller
# passing any of these in **fields would silently corrupt the
# invariants validate() depends on (most importantly, the monotonic
# seq counter) — emit() rejects that outright. See FIX for Finding 4.
_RESERVED_FIELDS = frozenset({"seq", "event", "run_id", "seed"})

# Order in which reserved fields are rebuilt when a record has to be
# reshaped. A tuple (not the frozenset above) so the reconstruction is
# deterministic; `seq` first because it is the one field that must
# never be altered in any way.
_RESERVED_FIELD_ORDER = ("seq", "event", "run_id", "seed")

# SINGLE SOURCE OF TRUTH for validate()'s per-event required fields.
# Both validate() (rules 2 and 3) and emit()'s budget fitter read this
# same mapping, so the "which fields must survive" knowledge cannot
# drift between the writer and the gate. Adding a rule here
# automatically (a) makes validate() enforce it and (b) makes emit()
# protect that field from truncation/dropping.
_REQUIRED_FIELDS_BY_EVENT: dict[str, tuple[str, ...]] = {
    "tool_call": ("name", "ok"),
    "model_call": ("prompt_tokens", "completion_tokens"),
}

# Per-line length ceiling for validate()'s JSON parse, guarding against
# JSON-bomb DoS input (pathologically deep nesting that can blow the
# JSON decoder's native stack before Python's own RecursionError
# machinery gets a chance to fire cleanly). See the module docstring's
# threat-model section for the empirical basis of this number.
_MAX_TRACE_LINE_CHARS = 100_000

# emit()'s write-side reshaping budget: comfortably (10,000 chars, 10%)
# below _MAX_TRACE_LINE_CHARS, so that Trace.emit can GUARANTEE — not
# merely hope — that every line it produces passes validate()'s length
# gate, regardless of how large, how long-named, or how numerous a
# caller's fields are. See "CONTRACT FOR TASK 2/3/4 AUTHORS".
_MAX_EMITTED_LINE_CHARS = 90_000

# How much slack to reserve, per truncated string, for the marker text
# itself ("...[truncated N chars]") — N can be several digits and JSON
# string-escaping can grow a handful of characters too. The fitter
# verifies the result rather than trusting this number, so it only has
# to be a good first guess.
_TRUNCATION_MARKER_SLACK = 40

# Ceiling on the SERIALIZED size of a protected (reserved or
# validate-required) value once a record has to be reshaped. Protected
# keys are always kept; only a pathologically large protected VALUE is
# shortened, and shortening it to this bound keeps the guaranteed-safe
# skeleton of the record tiny (<= ~8 keys x ~250 chars) so the
# last-resort fallback provably fits any sane budget.
_MAX_PROTECTED_VALUE_CHARS = 200

# Ceiling on the length of a non-protected field NAME. Longer names are
# shortened; the reserved/required names are all far below this.
_MAX_FIELD_NAME_CHARS = 200

# Ceiling on the number of non-protected fields carried on one event.
# Beyond this, surplus fields are dropped (and counted in
# `fields_dropped`) — 500 fields on a single trace event is already far
# past anything the harness legitimately logs, and an unbounded count
# is what lets "many small fields" defeat a value-only budget.
_MAX_EXTRA_FIELDS = 500


def _json_len(value) -> int:
    """Serialized length of one JSON value."""
    return len(json.dumps(value))


def _item_cost(key: str, value) -> int:
    """Upper bound on what one `key: value` pair costs inside a JSON
    object serialized by `json.dumps` with default separators.

    `json.dumps({"a": 1, "b": 2})` == `'{"a": 1, "b": 2}'`, so a pair
    costs `len(dumps(key)) + len(": ") + len(dumps(value))` plus the
    `", "` separator that precedes every pair but the first. Counting
    the separator unconditionally makes this a safe over-estimate.
    """
    return _json_len(key) + 2 + _json_len(value) + 2


def _shrink_value(value, budget: int):
    """Return a JSON value whose serialization is at most `budget`
    characters, or None if not even a minimal marker fits.

    Strings keep a prefix and gain a visible `"...[truncated N chars]"`
    suffix so the loss is obvious to anyone reading the trace (never
    silent corruption). Non-strings can't be sliced meaningfully, so an
    oversized one becomes a short marker naming its type and original
    serialized size. Purely a function of (value, budget) — no
    randomness, no wall-clock — so determinism holds.
    """
    if budget < 2:  # not even `""` fits
        return None

    if not isinstance(value, str):
        marker = (
            f"...[truncated {type(value).__name__} value, "
            f"~{_json_len(value)} chars serialized]"
        )
        if _json_len(marker) <= budget:
            return marker
        # Budget too tight even for the descriptive marker: fall
        # through to the string path, which shortens it further.
        value = marker

    original = len(value)
    keep = min(original, max(0, budget - _TRUNCATION_MARKER_SLACK))
    while True:
        removed = original - keep
        candidate = (
            value[:keep] + f"...[truncated {removed} chars]"
            if removed
            else value
        )
        excess = _json_len(candidate) - budget
        if excess <= 0:
            return candidate
        if keep == 0:
            break
        keep = max(0, keep - max(1, excess))

    # Even `"...[truncated N chars]"` is too long for this budget.
    for fallback in ("...[truncated]", "..."):
        if _json_len(fallback) <= budget:
            return fallback
    return ""


def _fit_record_to_budget(record: dict, budget: int) -> dict:
    """Return a record whose `json.dumps(sort_keys=True)` serialization
    is at most `budget` characters AND which still carries every field
    `Trace.validate` requires for its event kind.

    This is what makes `Trace.emit`'s guarantee to Task 2/3/4 authors
    structural rather than a convention someone has to remember: no
    matter what a caller passes — one huge value, twenty thousand tiny
    ones, a 200,000-character field NAME — the record that actually
    gets appended both fits the budget and passes the gate.

    Shape of the algorithm:

    1. Fast path: if the record already fits, return it untouched. This
       is the overwhelmingly common case and costs one `json.dumps`.
    2. Otherwise rebuild the record from a PROTECTED SKELETON: the
       reserved fields plus `_REQUIRED_FIELDS_BY_EVENT[event]` — the
       very same declaration `validate` reads. Protected values that
       are themselves enormous are shortened to
       `_MAX_PROTECTED_VALUE_CHARS`, but their keys are never dropped,
       which bounds the skeleton at roughly 2 KB.
    3. Spend the REMAINING budget on the other fields: surplus fields
       past `_MAX_EXTRA_FIELDS` are dropped (counted in
       `fields_dropped`), over-long names are shortened (de-duplicated
       with a `~N` suffix), and the rest are admitted smallest-cost
       first with an even share of what is left — so many small fields
       all survive while one giant field absorbs the truncation.
    4. Belt-and-braces: re-serialize; if the result somehow still
       exceeds the budget, fall back to the protected skeleton alone,
       which provably fits.

    Deterministic throughout: a pure function of `record` and `budget`,
    so `to_jsonl()`'s byte-identical guarantee still holds under
    reshaping.
    """
    if len(json.dumps(record, sort_keys=True)) <= budget:
        return record

    required = _REQUIRED_FIELDS_BY_EVENT.get(record.get("event"), ())
    protected_keys = (*_RESERVED_FIELD_ORDER, *required)

    # --- Step 2: the protected skeleton -----------------------------
    skeleton: dict = {}
    for key in protected_keys:
        if key not in record:
            # A required field the caller never passed. Not emit's job
            # to invent one — validate will fail with the TRUE reason
            # ("tool_call missing field: name"), which is exactly the
            # feedback the caller needs.
            continue
        value = record[key]
        # `seq` is emit-owned, always a small int, and load-bearing for
        # validate's monotonicity rule — never reshape it.
        if key != "seq" and _json_len(value) > _MAX_PROTECTED_VALUE_CHARS:
            shrunk = _shrink_value(value, _MAX_PROTECTED_VALUE_CHARS)
            value = "" if shrunk is None else shrunk
        skeleton[key] = value

    # --- Step 3: the remaining fields -------------------------------
    extras = [
        (key, value)
        for key, value in record.items()
        if key not in skeleton and key not in ("fields_truncated", "fields_dropped")
    ]
    dropped = 0
    if len(extras) > _MAX_EXTRA_FIELDS:
        dropped = len(extras) - _MAX_EXTRA_FIELDS
        extras = extras[:_MAX_EXTRA_FIELDS]

    skeleton["fields_truncated"] = True
    if dropped:
        skeleton["fields_dropped"] = dropped

    remaining = budget - 2 - sum(
        _item_cost(k, v) for k, v in skeleton.items()
    )

    # Shorten over-long names, de-duplicating any collisions the
    # shortening introduces (two 200k-char names sharing a prefix).
    seen = set(skeleton)
    renamed: list[tuple[str, object]] = []
    for key, value in extras:
        if len(key) > _MAX_FIELD_NAME_CHARS:
            key = key[:_MAX_FIELD_NAME_CHARS] + "...[truncated name]"
        if key in seen:
            suffix = 2
            while f"{key}~{suffix}" in seen:
                suffix += 1
            key = f"{key}~{suffix}"
        seen.add(key)
        renamed.append((key, value))

    # Cheapest first, so the budget left over from small fields flows
    # to the large one(s). Ties broken by key for determinism.
    entries = sorted(
        ((_item_cost(k, v), k, v) for k, v in renamed),
        key=lambda entry: (entry[0], entry[1]),
    )

    fitted = dict(skeleton)
    left = len(entries)
    for cost, key, value in entries:
        share = remaining // left if left > 0 else 0
        left -= 1
        if cost <= share:
            fitted[key] = value
            remaining -= cost
            continue
        overhead = _json_len(key) + 4  # `": "` plus the `", "` separator
        shrunk = _shrink_value(value, share - overhead)
        if shrunk is None:
            continue  # not even a marker fits; drop this field
        actual = overhead + _json_len(shrunk)
        if actual > remaining:
            continue
        fitted[key] = shrunk
        remaining -= actual

    # --- Step 4: belt-and-braces ------------------------------------
    # _item_cost over-estimates, so this should never fire; if a future
    # change makes the accounting wrong, degrade to the protected
    # skeleton (~2 KB) rather than emit a line the gate would reject.
    if len(json.dumps(fitted, sort_keys=True)) > budget:
        return skeleton

    return fitted


class Trace:
    """Accumulates JSONL trace events for a single agent run.

    `run_id` and `seed` are stamped onto every event so a trace can be
    replayed or attributed without external bookkeeping. Determinism is
    load-bearing for this lab's leaderboard: given the same seed and
    the same sequence of `emit()` calls, `to_jsonl()` must produce
    byte-identical output every time. `Trace` itself introduces no
    non-determinism (no wall-clock timestamps, no randomness) — that
    guarantee holds as long as callers don't pass non-deterministic
    values in `**fields`.
    """

    def __init__(self, run_id: str, seed: int) -> None:
        self.run_id = run_id
        self.seed = seed
        self._events: list[dict] = []
        self._seq = 0

    def emit(self, event: str, **fields) -> None:
        """Append one event to the trace.

        Raises ValueError if `event` is not one of `EVENTS`, or if
        `fields` tries to override the reserved keys `seq`, `run_id`,
        or `seed` — those are always stamped by `Trace` itself so the
        monotonic-`seq` invariant `validate()` depends on can never be
        clobbered by a caller (e.g. a middleware layer logging a
        tool's own sequence number under the name `seq`). Passing
        `event` a second time via `fields` is also rejected, but as a
        plain Python `TypeError` ("got multiple values for argument
        'event'") raised by ordinary keyword-argument binding before
        this method's body even runs — not the `ValueError` the other
        three reserved keys raise. Either way, the override never
        takes effect.

        Otherwise deliberately does NOT reject `fields` for content or
        size. Content is `validate`'s job (run once over the finished
        JSONL, not a per-call tax during agent execution). Size, field
        count and field-name length are handled by automatic
        RESHAPING: if the record's JSON serialization would exceed
        `_MAX_EMITTED_LINE_CHARS`, values are truncated (with a visible
        marker), over-long names are shortened, and surplus fields are
        dropped — but the reserved fields and the fields
        `_REQUIRED_FIELDS_BY_EVENT` marks as required for this event
        kind are always preserved. So a call that supplies the required
        fields can never produce a line `validate` rejects, whatever
        else it carries. See the module docstring's "CONTRACT FOR TASK
        2/3/4 AUTHORS" for exactly what may be lost (and why you should
        keep your own `doc_id`-style reference to very large content).
        """
        if event not in EVENTS:
            raise ValueError(
                f"unknown event {event!r}; must be one of {EVENTS}"
            )
        clobbered = _RESERVED_FIELDS & fields.keys()
        if clobbered:
            raise ValueError(
                f"fields cannot override reserved keys: {sorted(clobbered)}"
            )
        record = {
            "seq": self._seq,
            "event": event,
            "run_id": self.run_id,
            "seed": self.seed,
            **fields,
        }
        record = _fit_record_to_budget(record, _MAX_EMITTED_LINE_CHARS)
        self._events.append(record)
        self._seq += 1

    def to_jsonl(self) -> str:
        """Render the recorded events as newline-delimited JSON.

        Keys within each record are sorted for byte-identical output
        across runs (dict insertion order alone would already be
        deterministic here, but sorting also protects against any
        future caller that builds `fields` from an unordered source,
        e.g. a set or a dict comprehension over unordered keys).
        """
        return "\n".join(
            json.dumps(record, sort_keys=True) for record in self._events
        )

    @staticmethod
    def validate(jsonl: str) -> tuple[bool, str]:
        """Trace-conformance GATE: PASS/FAIL, never a scored dimension.

        Returns (True, "") only if all of the following hold:
          1. events, in `seq` order, start with `agent_start` and end
             with `agent_end` (this both requires each to be present
             AND requires them in the right relative position — a
             trace where `agent_end` precedes `agent_start` fails)
          2. every `tool_call` event carries both `name` and `ok`
          3. every `model_call` event carries both `prompt_tokens` and
             `completion_tokens`
          4. events are ordered by a strictly monotonically increasing
             integer `seq` field (non-int `seq`, e.g. a float or a
             string, fails — see the int-only decision in the module
             docstring)

        Rules 2 and 3 are not written out longhand here: they are read
        from `_REQUIRED_FIELDS_BY_EVENT`, the same declaration
        `Trace.emit`'s budget fitter uses to decide which fields it must
        never drop. One declaration, two readers — the gate and the
        writer cannot drift apart.

        Additionally, every event's `event` field must be one of
        `EVENTS` — an unrecognized event name (e.g. a typo like
        `"tool_calll"`) fails outright rather than silently skipping
        the field checks that would otherwise apply to it. This closes
        the hole where renaming an event lets it bypass rules 2/3.

        This method is designed to never raise. Any input that isn't
        well-formed JSONL of JSON objects — empty input, non-JSON
        lines, lines that parse to something other than a JSON object
        (e.g. `null`, `42`, `[]`), wrong-typed `seq`, an oversized or
        pathologically nested line crafted as a JSON-bomb DoS, etc. —
        is reported as `(False, "<specific reason>")`, not an
        exception. `validate` is the gate that decides whether a
        student's run is scored at all, and it must handle
        hand-written/adversarial JSONL — including JSON bombs — as an
        expected input, not a hypothetical. See the module docstring
        for the two defenses (a line-length ceiling, and a broad
        `except Exception` around the JSON parse) and why both are
        needed.

        On failure, `reason` is a short, machine-greppable string naming
        the specific rule that failed (e.g. "missing agent_end",
        "tool_call missing field: name") — never a generic "invalid",
        and never a reason that misdescribes what actually went wrong.
        """
        lines = [line for line in jsonl.splitlines() if line.strip()]

        if not lines:
            return False, "missing agent_start: trace is empty"

        records: list[dict] = []
        for i, line in enumerate(lines):
            # Defense 1: reject grossly oversized lines before ever
            # attempting to parse them — cheap, and closes off the
            # obvious JSON-bomb shape (a single line with hundreds of
            # thousands of characters of nested brackets) without
            # spending any parse work on it. See _MAX_TRACE_LINE_CHARS
            # and the module docstring for the chosen limit and why it
            # can't fail an honest student (emit() structurally cannot
            # produce a line this long).
            if len(line) > _MAX_TRACE_LINE_CHARS:
                return (
                    False,
                    f"line {i} exceeds max trace line length "
                    f"({_MAX_TRACE_LINE_CHARS} chars): refusing to "
                    f"parse (possible JSON bomb)",
                )
            # Defense 2: a bare `except Exception`, not just
            # `except json.JSONDecodeError`. A hand-crafted line that
            # slips under the length ceiling above but is still deeply
            # nested can make the decoder raise something other than
            # JSONDecodeError — observed: RecursionError from a native
            # stack overflow inside the C JSON decoder, which is not a
            # JSONDecodeError and is not reliably preventable by tuning
            # sys.setrecursionlimit (the C decoder's stack check is
            # independent of Python's own recursion-limit machinery).
            # At this boundary, the never-raise invariant outranks
            # exception-type precision, so we deliberately catch
            # anything json.loads can throw.
            try:
                parsed = json.loads(line)
            except Exception as exc:
                return (
                    False,
                    f"malformed json on line {i}: "
                    f"{type(exc).__name__}: {exc}",
                )
            if not isinstance(parsed, dict):
                return (
                    False,
                    f"line {i} is not a JSON object (got "
                    f"{type(parsed).__name__}): {parsed!r}",
                )
            records.append(parsed)

        # Every record must name a recognized event. Checked before any
        # rule-specific field checks so a misspelled/unknown event name
        # can't dodge the tool_call/model_call field checks below
        # (Finding 3).
        for i, record in enumerate(records):
            event = record.get("event")
            if event is None:
                return False, f"missing event field on line {i}"
            if event not in EVENTS:
                return (
                    False,
                    f"unknown event {event!r} on line {i}; must be one "
                    f"of {EVENTS}",
                )

        # Rule 4: seq must be present, an int (not bool, not float, not
        # str — see the int-only decision in the module docstring), and
        # strictly monotonically increasing. Type-checked before any
        # comparison so mismatched/non-orderable seq values can never
        # reach a `>` comparison and raise (Finding 1).
        prev_seq = None
        for i, record in enumerate(records):
            if "seq" not in record:
                return False, f"missing seq field on line {i}"
            seq = record["seq"]
            if not isinstance(seq, int) or isinstance(seq, bool):
                return (
                    False,
                    f"seq must be an int on line {i}, got "
                    f"{type(seq).__name__}: {seq!r}",
                )
            if prev_seq is not None and not (seq > prev_seq):
                return (
                    False,
                    f"seq not monotonically increasing at line {i}: "
                    f"{seq!r} follows {prev_seq!r}",
                )
            prev_seq = seq

        # Rule 1: in seq order (== record order, since seq is now known
        # to be strictly increasing across `records`), the trace must
        # start with agent_start and end with agent_end. This both
        # requires presence of each AND requires them in the right
        # relative order (Finding 2) — e.g. agent_end followed later by
        # agent_start fails here because the first record isn't
        # agent_start.
        if records[0]["event"] != "agent_start":
            return (
                False,
                f"missing agent_start: first event is "
                f"{records[0]['event']!r}, not agent_start",
            )
        if records[-1]["event"] != "agent_end":
            return (
                False,
                f"missing agent_end: last event is "
                f"{records[-1]['event']!r}, not agent_end",
            )

        # Rules 2 and 3: per-event required fields, read from the
        # single source of truth shared with emit()'s budget fitter.
        for i, record in enumerate(records):
            for field in _REQUIRED_FIELDS_BY_EVENT.get(record["event"], ()):
                if field not in record:
                    return (
                        False,
                        f"{record['event']} missing field: {field} "
                        f"(line {i})",
                    )

        return True, ""
