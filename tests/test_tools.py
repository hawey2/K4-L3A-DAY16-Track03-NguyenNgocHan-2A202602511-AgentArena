"""Tests for arena.tools — the tool surface with seeded hostility."""

import json
import time

import pytest

from arena.corpus import Corpus
from arena.tools import Tools, ToolResult
from arena.trace import Trace

_CORPUS = Corpus.generate(seed=42)  # shared, read-only fixture-ish corpus


def _make_tools(seed=42, flaky=True):
    trace = Trace(run_id="test-run", seed=seed)
    tools = Tools(corpus=_CORPUS, trace=trace, seed=seed, flaky=flaky)
    return trace, tools


def _events(trace: Trace) -> list[dict]:
    jsonl = trace.to_jsonl()
    return [json.loads(line) for line in jsonl.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Step 1 (TDD): fetch_doc returns the body and emits a tool_call event.
# ---------------------------------------------------------------------------


def test_fetch_doc_returns_body_and_emits_tool_call():
    known_id = sorted(_CORPUS.doc_ids())[0]
    trace, tools = _make_tools(flaky=False)

    result = tools.fetch_doc(known_id)

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.content == _CORPUS.get(known_id).body

    tool_calls = [e for e in _events(trace) if e["event"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "fetch_doc"
    assert tool_calls[0]["ok"] is True
    assert tool_calls[0]["doc_id"] == known_id


def test_fetch_doc_unknown_id_returns_ok_false_with_error():
    _, tools = _make_tools(flaky=False)
    result = tools.fetch_doc("doc-does-not-exist-9999")
    assert result.ok is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# Step 2 (TDD): flakiness determinism over 50 calls.
# ---------------------------------------------------------------------------


def test_flakiness_is_deterministic_per_seed_over_50_calls():
    known_id = sorted(_CORPUS.doc_ids())[0]

    _, tools_a1 = _make_tools(seed=42, flaky=True)
    _, tools_a2 = _make_tools(seed=42, flaky=True)
    _, tools_b = _make_tools(seed=7, flaky=True)

    seq_a1 = [tools_a1.fetch_doc(known_id).ok for _ in range(50)]
    seq_a2 = [tools_a2.fetch_doc(known_id).ok for _ in range(50)]
    seq_b = [tools_b.fetch_doc(known_id).ok for _ in range(50)]

    assert seq_a1 == seq_a2, "same seed must produce the same pass/fail sequence"
    assert seq_a1 != seq_b, "a different seed should produce a different sequence"
    assert any(ok is False for ok in seq_a1), "expected >=1 flaky failure in 50 calls at 15% rate"
    assert any(ok is True for ok in seq_a1), "expected >=1 success in 50 calls"


def test_flaky_false_never_fails():
    known_id = sorted(_CORPUS.doc_ids())[0]
    _, tools = _make_tools(flaky=False)
    for _ in range(50):
        assert tools.fetch_doc(known_id).ok is True


def test_flaky_roll_boolean_is_the_same_regardless_of_which_tool_is_called():
    """Whether call index `idx` is flaky AT ALL is a pure function of
    (seed, idx), independent of which tool is invoked — only the
    *content* of the perturbation (which of timeout/truncate/noise,
    and whether that even changes `ok`) is tool-specific: e.g. a
    "truncate" outcome on fetch_doc still reports ok=True, while
    calc's only flaky mode is "timeout" (always ok=False). So this
    checks the underlying roll via the trace's `flaky_mode` field
    rather than `ok`, which the other test already covers for a
    single tool."""
    known_id = sorted(_CORPUS.doc_ids())[0]
    trace_fetch, tools_fetch = _make_tools(seed=99, flaky=True)
    trace_calc, tools_calc = _make_tools(seed=99, flaky=True)

    for _ in range(50):
        tools_fetch.fetch_doc(known_id)
        tools_calc.calc("1+1")

    was_flaky_fetch = [
        e["flaky_mode"] is not None for e in _events(trace_fetch) if e["event"] == "tool_call"
    ]
    was_flaky_calc = [
        e["flaky_mode"] is not None for e in _events(trace_calc) if e["event"] == "tool_call"
    ]
    assert was_flaky_fetch == was_flaky_calc


# ---------------------------------------------------------------------------
# Step 3 (TDD): calc safety — never eval(), blocks __import__.
# ---------------------------------------------------------------------------


def test_calc_basic_arithmetic():
    _, tools = _make_tools(flaky=False)
    result = tools.calc("2 + 3 * 4")
    assert result.ok is True
    assert result.content == "14"


def test_calc_does_not_use_eval_and_blocks_import():
    _, tools = _make_tools(flaky=False)
    result = tools.calc('__import__("os")')
    assert result.ok is False
    assert result.error is not None


def test_calc_blocks_attribute_and_call_access():
    _, tools = _make_tools(flaky=False)
    for expr in ["().__class__", "open('x')", "[].__class__.__bases__"]:
        result = tools.calc(expr)
        assert result.ok is False, f"expected {expr!r} to be rejected"


def test_calc_rejects_absurd_exponent_without_hanging():
    _, tools = _make_tools(flaky=False)
    result = tools.calc("9 ** 99999999")
    assert result.ok is False


def test_calc_rejects_mult_chain_dos_without_hanging():
    """fix-round-1 finding 1 (CRITICAL): a Mult chain built from nested,
    duplicated parens — `(x)*(x)` doubling — has NO `**` operator
    anywhere, so a guard that only bounded `**` let it sail through and
    hang the process on a multi-million-digit int before an eventual
    ok=False fell out of CPython's own int-to-str digit limit (not a
    defence this module added). The fix bounds every intermediate
    value, not just the result of `**`, so this must now reject FAST —
    well under a second, not merely "eventually"."""
    expr = "9"
    for _ in range(20):  # value (and text) roughly doubles each level
        expr = f"({expr})*({expr})"
    assert len(expr) > 100_000, "test setup: expression should already be huge"

    _, tools = _make_tools(flaky=False)
    start = time.perf_counter()
    result = tools.calc(expr)
    elapsed = time.perf_counter() - start

    assert result.ok is False
    assert elapsed < 1.0, f"calc took {elapsed:.3f}s — DoS guard did not fire early"


def test_calc_rejects_long_linear_chain_without_hanging_but_allows_reasonable_length():
    """A merely-long (not exponentially-duplicated) expression under
    the length cap should still evaluate normally and fast."""
    _, tools = _make_tools(flaky=False)
    expr = "+".join(["1"] * 500)  # 999 chars, well under the 2000 cap
    start = time.perf_counter()
    result = tools.calc(expr)
    elapsed = time.perf_counter() - start
    assert result.ok is True
    assert result.content == "500"
    assert elapsed < 1.0


def test_calc_rejects_overlong_expression_before_parsing():
    _, tools = _make_tools(flaky=False)
    expr = "1+" * 2000 + "1"  # far past the 2000-char cap
    result = tools.calc(expr)
    assert result.ok is False


def test_calc_rejects_infinite_and_nan_results():
    """fix-round-1 finding 4 (Minor): a calculator reporting inf/nan as
    a successful, submittable result is wrong — must be ok=False."""
    _, tools = _make_tools(flaky=False)
    for expr in ("1e309", "1e309 - 1e309", "1e200 * 1e200"):
        result = tools.calc(expr)
        assert result.ok is False, f"expected {expr!r} to be rejected"
        assert result.content not in ("inf", "nan", "-inf")


def test_calc_division_by_zero_is_ok_false_not_a_raised_exception():
    _, tools = _make_tools(flaky=False)
    result = tools.calc("1/0")
    assert result.ok is False


def test_calc_syntax_error_is_ok_false():
    _, tools = _make_tools(flaky=False)
    result = tools.calc("2 + * 3")
    assert result.ok is False


# ---------------------------------------------------------------------------
# search / submit / calls counter
# ---------------------------------------------------------------------------


def test_search_returns_tool_result_with_json_content():
    _, tools = _make_tools(flaky=False)
    result = tools.search("chính sách", k=3)
    assert result.ok is True
    payload = json.loads(result.content)
    assert isinstance(payload, list)
    assert len(payload) <= 3
    for item in payload:
        assert "doc_id" in item and "title" in item


def test_calls_counter_increments_per_call_across_methods():
    _, tools = _make_tools(flaky=False)
    known_id = sorted(_CORPUS.doc_ids())[0]
    assert tools.calls == 0
    tools.search("chính sách")
    assert tools.calls == 1
    tools.fetch_doc(known_id)
    assert tools.calls == 2
    tools.calc("1+1")
    assert tools.calls == 3
    tools.submit({"answer": "x"})
    assert tools.calls == 4


def test_submit_accepts_dict_report_and_trace_is_not_aliased():
    trace, tools = _make_tools(flaky=False)
    report = {"answer": "42", "citations": ["doc-0001"]}

    result = tools.submit(report)
    assert result.ok is True

    # mutate the caller's dict AFTER submit — the trace must not change
    # retroactively (see arena/trace.py's aliasing warning).
    report["answer"] = "MUTATED"
    report["citations"].append("doc-9999")

    submit_events = [e for e in _events(trace) if e["event"] == "tool_call" and e["name"] == "submit"]
    assert submit_events
    assert "MUTATED" not in submit_events[0]["report_json"]
    assert "doc-9999" not in submit_events[0]["report_json"]


def test_submit_rejects_non_dict_report():
    _, tools = _make_tools(flaky=False)
    result = tools.submit("not a dict")  # type: ignore[arg-type]
    assert result.ok is False


def test_every_tool_call_emits_required_trace_fields():
    trace, tools = _make_tools(flaky=False)
    known_id = sorted(_CORPUS.doc_ids())[0]
    tools.search("chính sách")
    tools.fetch_doc(known_id)
    tools.calc("1+1")
    tools.submit({"answer": "x"})

    tool_calls = [e for e in _events(trace) if e["event"] == "tool_call"]
    assert len(tool_calls) == 4
    for e in tool_calls:
        assert "name" in e and "ok" in e


# ---------------------------------------------------------------------------
# Full session -> Trace.validate (TDD order's final checklist item)
# ---------------------------------------------------------------------------


def test_full_tool_session_produces_a_trace_that_passes_validate():
    trace, tools = _make_tools(seed=42, flaky=True)
    known_id = sorted(_CORPUS.doc_ids())[0]

    trace.emit("agent_start", brief_id="b-test")
    trace.emit("model_call", prompt_tokens=42, completion_tokens=17)
    tools.search("chính sách nghỉ phép")
    tools.fetch_doc(known_id)
    tools.calc("21 * 2")
    tools.submit({"answer": "demo"})
    trace.emit("agent_end", status="ok")

    ok, reason = Trace.validate(trace.to_jsonl())
    assert ok is True, reason


def test_full_tool_session_passes_validate_even_when_every_call_is_flaky():
    """Sweep a range of seeds so this exercises timeout/truncate/noise
    outcomes too, not just whichever branch a single fixed seed hits —
    Trace.validate must PASS regardless of tool outcome content."""
    known_id = sorted(_CORPUS.doc_ids())[0]
    for seed in range(20):
        trace, tools = _make_tools(seed=seed, flaky=True)
        trace.emit("agent_start", brief_id="sweep")
        tools.search("chính sách")
        tools.fetch_doc(known_id)
        tools.calc("1+1")
        tools.submit({"answer": "x"})
        trace.emit("agent_end", status="ok")

        ok, reason = Trace.validate(trace.to_jsonl())
        assert ok is True, f"seed={seed}: {reason}"
