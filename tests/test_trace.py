"""Tests for arena.trace — the frozen trace schema and conformance gate.

These tests double as the executable spec for later tasks: the scorer,
the harness layers, and the leaderboard all depend on Trace.validate
behaving exactly as asserted here.
"""

import json
import time

import pytest

from arena.trace import EVENTS, Trace


# ---------------------------------------------------------------------------
# Step 1 (TDD): the exact failing test from the brief.
# ---------------------------------------------------------------------------


def test_validate_rejects_trace_without_agent_end():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is False and "agent_end" in reason


# ---------------------------------------------------------------------------
# One test per validate rule, plus the happy path.
# ---------------------------------------------------------------------------


def test_validate_rejects_trace_without_agent_start():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is False
    assert "agent_start" in reason


def test_validate_rejects_tool_call_missing_name():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("tool_call", ok=True)  # missing "name"
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is False
    assert "tool_call" in reason and "name" in reason


def test_validate_rejects_tool_call_missing_ok():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("tool_call", name="search")  # missing "ok"
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is False
    assert "tool_call" in reason and "ok" in reason


def test_validate_rejects_model_call_missing_prompt_tokens():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("model_call", completion_tokens=10)  # missing prompt_tokens
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is False
    assert "model_call" in reason and "prompt_tokens" in reason


def test_validate_rejects_model_call_missing_completion_tokens():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("model_call", prompt_tokens=10)  # missing completion_tokens
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is False
    assert "model_call" in reason and "completion_tokens" in reason


def test_validate_rejects_non_monotonic_seq():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("agent_end", status="ok")
    lines = t.to_jsonl().splitlines()
    # Tamper: duplicate the seq of the first line onto the second, breaking
    # strict monotonicity.
    import json

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    second["seq"] = first["seq"]
    tampered = "\n".join([lines[0], json.dumps(second)])
    ok, reason = Trace.validate(tampered)
    assert ok is False
    assert "seq" in reason


def test_validate_happy_path_returns_true_and_empty_reason():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("model_call", prompt_tokens=100, completion_tokens=20)
    t.emit("tool_call", name="search", ok=True)
    t.emit("layer", name="critic", note="looks fine")
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is True
    assert reason == ""


def test_emit_rejects_unknown_event():
    t = Trace(run_id="r1", seed=7)
    try:
        t.emit("not_a_real_event")
    except ValueError as exc:
        assert "not_a_real_event" in str(exc)
    else:
        raise AssertionError("emit() should reject events outside EVENTS")


def test_events_tuple_matches_contract():
    assert EVENTS == (
        "agent_start",
        "model_call",
        "tool_call",
        "layer",
        "agent_end",
    )


# ---------------------------------------------------------------------------
# Determinism: identical seed + identical emit sequence => identical JSONL.
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_emits_byte_identical_output():
    def build():
        t = Trace(run_id="r1", seed=42)
        t.emit("agent_start", brief_id="b1")
        t.emit("model_call", prompt_tokens=50, completion_tokens=5)
        t.emit("tool_call", name="fetch", ok=True)
        t.emit("agent_end", status="ok")
        return t.to_jsonl()

    a = build()
    b = build()
    assert a == b


# ---------------------------------------------------------------------------
# Fix round 1 — adversarial coverage.
#
# validate() is a @staticmethod that accepts arbitrary text: students are
# motivated to hand-write JSONL to game the trace-conformance gate, so
# adversarial/malformed input is an EXPECTED input, not a hypothetical.
# validate() must NEVER raise; every case below must come back as a normal
# (False, reason) tuple with a specific, greppable reason.
# ---------------------------------------------------------------------------


# Finding 1 (CRITICAL): validate() raised on several shapes of adversarial
# input instead of returning (False, reason). Each case below reproduces one
# of the reviewer's repros and asserts it now degrades gracefully.


def test_validate_empty_string_does_not_raise():
    ok, reason = Trace.validate("")
    assert ok is False
    assert reason


def test_validate_whitespace_only_does_not_raise():
    ok, reason = Trace.validate("   \n\n   \n")
    assert ok is False
    assert reason


def test_validate_bare_null_does_not_raise():
    # Reviewer repro: Trace.validate("null") used to raise TypeError
    # ("argument of type 'NoneType' is not a container").
    ok, reason = Trace.validate("null")
    assert ok is False
    assert "not a JSON object" in reason


def test_validate_bare_json_array_does_not_raise():
    ok, reason = Trace.validate("[]")
    assert ok is False
    assert "not a JSON object" in reason


def test_validate_pretty_printed_json_object_does_not_raise():
    # A single JSON object (not JSONL) pretty-printed across several
    # physical lines: most lines are not valid standalone JSON on their
    # own, so this must fail cleanly with a specific per-line reason
    # rather than crash.
    pretty = json.dumps({"seq": 0, "event": "agent_start"}, indent=2)
    assert "\n" in pretty  # sanity check this really spans multiple lines
    ok, reason = Trace.validate(pretty)
    assert ok is False
    assert reason


def test_validate_single_line_json_object_is_treated_as_one_event():
    # The whole trace is exactly one JSON object on one line (not multiple
    # JSONL records). This is well-formed input for validate() — it parses
    # as a single record — and correctly fails on missing agent_end rather
    # than crashing.
    ok, reason = Trace.validate('{"seq": 0, "event": "agent_start"}')
    assert ok is False
    assert "agent_end" in reason


def test_validate_malformed_json_among_valid_lines_does_not_raise():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("agent_end", status="ok")
    lines = t.to_jsonl().splitlines()
    tampered = "\n".join([lines[0], "{not valid json,,,", lines[1]])
    ok, reason = Trace.validate(tampered)
    assert ok is False
    assert "malformed json" in reason


def test_validate_bare_scalar_line_does_not_raise():
    # Reviewer repro: a line that parses to a bare scalar (int) rather
    # than a JSON object used to raise TypeError when probed for "seq".
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("agent_end", status="ok")
    lines = t.to_jsonl().splitlines()
    tampered = "\n".join([lines[0], "42", lines[1]])
    ok, reason = Trace.validate(tampered)
    assert ok is False
    assert "not a JSON object" in reason


def test_validate_non_integer_seq_does_not_raise():
    # Reviewer repro: a str seq used to raise TypeError when compared
    # with '>' against an int seq.
    ok, reason = Trace.validate(
        '{"seq": "abc", "event": "agent_start"}\n'
        '{"seq": 1, "event": "agent_end"}'
    )
    assert ok is False
    assert "seq must be an int" in reason


def test_validate_mixed_type_seq_does_not_raise():
    ok, reason = Trace.validate(
        '{"seq": 0, "event": "agent_start"}\n'
        '{"seq": "abc", "event": "agent_end"}'
    )
    assert ok is False
    assert "seq must be an int" in reason


def test_validate_float_seq_rejected():
    # seq decision: int-only (see module docstring). A float seq — even
    # one that looks orderable, e.g. 0, then 0.5 — is rejected.
    ok, reason = Trace.validate(
        '{"seq": 0, "event": "agent_start"}\n'
        '{"seq": 0.5, "event": "agent_end"}'
    )
    assert ok is False
    assert "seq must be an int" in reason


def test_validate_duplicate_seq_rejected():
    ok, reason = Trace.validate(
        '{"seq": 0, "event": "agent_start"}\n'
        '{"seq": 0, "event": "agent_end"}'
    )
    assert ok is False
    assert "seq" in reason


def test_validate_backwards_seq_rejected():
    ok, reason = Trace.validate(
        '{"seq": 1, "event": "agent_start"}\n'
        '{"seq": 0, "event": "agent_end"}'
    )
    assert ok is False
    assert "seq" in reason


# Finding 2 (Important): agent_end before agent_start used to pass because
# rule 1 only checked presence, not order.


def test_validate_rejects_agent_end_before_agent_start():
    ok, reason = Trace.validate(
        '{"seq": 0, "event": "agent_end"}\n'
        '{"seq": 1, "event": "agent_start"}'
    )
    assert ok is False
    assert "agent_start" in reason


# Finding 3 (Important): unknown/misspelled event names used to bypass the
# tool_call/model_call field checks entirely.


def test_validate_rejects_unknown_event_name():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    lines = t.to_jsonl().splitlines()
    # Inject a misspelled tool_call event with none of the required
    # fields — under the old code this would silently skip rule 2.
    forged = json.dumps({"seq": 1, "event": "tool_calll"})
    t.emit("agent_end", status="ok")
    end_line = t.to_jsonl().splitlines()[-1]
    tampered = "\n".join([lines[0], forged, end_line])
    ok, reason = Trace.validate(tampered)
    assert ok is False
    assert "unknown event" in reason
    assert "tool_calll" in reason


# Finding 4 (Important): emit() let **fields silently clobber the reserved
# keys seq/event/run_id/seed, corrupting the invariants validate() depends
# on (most importantly the monotonic seq counter).


@pytest.mark.parametrize("reserved_key", ["seq", "event", "run_id", "seed"])
def test_emit_rejects_reserved_field_override(reserved_key):
    t = Trace(run_id="r1", seed=7)
    # "event" is also emit()'s own positional parameter name, so passing
    # it a second time via **fields collides at the Python call-binding
    # level (TypeError) before Trace's own reserved-field check ever
    # runs; seq/run_id/seed go through **fields cleanly and hit our
    # explicit ValueError. Either way the override must be rejected, not
    # silently applied.
    with pytest.raises((TypeError, ValueError), match=reserved_key):
        t.emit("agent_start", **{reserved_key: "clobbered"})


def test_emit_reserved_field_rejection_does_not_corrupt_seq_counter():
    # An honest emit() call must still work, and the seq counter must be
    # unaffected by a rejected call in between.
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    try:
        t.emit("model_call", seq=999, prompt_tokens=1, completion_tokens=1)
    except ValueError:
        pass
    t.emit("model_call", prompt_tokens=1, completion_tokens=1)
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is True and reason == ""


# ---------------------------------------------------------------------------
# Fix round 2 — Finding 1 remained open: validate() raised RecursionError
# on a JSON-bomb line (pathological nesting depth) instead of returning
# (False, reason). The re-reviewer's repro used array nesting; object
# nesting is checked too, since the recursion is in the JSON decoder's
# generic value-parsing path, not something array-specific.
# ---------------------------------------------------------------------------


# The exact bomb shape from the re-review: a single line with 200,000
# levels of array nesting inside an otherwise well-formed event.
_ARRAY_BOMB_LINE = (
    '{"seq":0,"event":"agent_start","payload":'
    + "[" * 200_000
    + "]" * 200_000
    + "}"
)

# Same idea, object nesting instead of array nesting.
_OBJECT_BOMB_LINE = (
    '{"seq":0,"event":"agent_start","payload":'
    + '{"a":' * 200_000
    + "1"
    + "}" * 200_000
    + "}"
)


def test_validate_array_nesting_bomb_does_not_raise():
    ok, reason = Trace.validate(_ARRAY_BOMB_LINE)
    assert ok is False
    assert isinstance(reason, str) and reason


def test_validate_object_nesting_bomb_does_not_raise():
    ok, reason = Trace.validate(_OBJECT_BOMB_LINE)
    assert ok is False
    assert isinstance(reason, str) and reason


def test_validate_line_length_ceiling_rejects_bomb_before_parsing():
    # Both bomb lines are ~400KB, comfortably over the 100,000-char
    # ceiling — confirm they're rejected specifically by the length
    # check (fast, no parse attempted), not by some other path.
    for bomb in (_ARRAY_BOMB_LINE, _OBJECT_BOMB_LINE):
        ok, reason = Trace.validate(bomb)
        assert ok is False
        assert "exceeds max trace line length" in reason
        assert "JSON bomb" in reason


# ---------------------------------------------------------------------------
# Fix round 3 — Important: the length ceiling could false-positive-reject an
# HONEST student. A legitimate tool_call carrying a fetched document body
# (t.emit("tool_call", name="fetch_doc", ok=True, result=<100K chars>)) came
# to 100,106 characters once wrapped in the JSON record — over the 100,000
# ceiling — even though nothing about it was adversarial. Since Corpus
# (Task 2) and Tools/ToolResult (Task 3) don't exist yet, nothing bounds
# what a legitimate tool result can contain today, and emit() deliberately
# doesn't reject fields for size. Fix: emit() now truncates oversized field
# values itself (see _fit_record_to_budget / _MAX_EMITTED_LINE_CHARS in
# arena/trace.py), so it is structurally impossible — not just unlikely —
# for Trace.emit to produce a line validate()'s ceiling would reject.
#
# The previous version of this test used a 5,000-char note field (5% of
# the ceiling) and gave false confidence: it could not have caught this
# class of failure. Replaced with a realistic 50-80KB document-body-sized
# field, plus the reviewer's exact reported scenario, plus a
# property-style sweep across field sizes including well over the ceiling.
# ---------------------------------------------------------------------------


def test_validate_realistic_large_document_field_validates_cleanly():
    # A plausible fetched-document body (50-80KB, the size the review
    # flagged as realistic) must validate cleanly. At this size it fits
    # under emit()'s _MAX_EMITTED_LINE_CHARS budget entirely untouched
    # (verified below), so nothing is lost — but the test's contract is
    # just "validates cleanly," which holds either way.
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    doc = "The quick brown fox. " * 3_500  # ~77,000 chars
    assert 50_000 <= len(doc) <= 80_000
    t.emit("tool_call", name="fetch_doc", ok=True, result=doc)
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is True and reason == ""

    # A document in this realistic size range should pass through
    # untouched — truncation is reserved for values that actually
    # threaten the budget, not ordinary large-but-reasonable content.
    events = t.to_jsonl().splitlines()
    import json

    tool_call_record = json.loads(events[1])
    assert tool_call_record["result"] == doc


def test_validate_reviewer_repro_100k_char_field_validates_cleanly():
    # The exact scenario from the fix-round-3 review: a 100,000-char tool
    # result, which (once wrapped in the JSON record with seq/event/
    # run_id/seed/name/ok) would be 100,106 chars — over the old, naive
    # ceiling — must now validate cleanly because emit() truncates it to
    # fit before it's ever handed to validate().
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    big_doc = "x" * 100_000
    t.emit("tool_call", name="fetch_doc", ok=True, result=big_doc)
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is True and reason == ""


def test_emit_truncates_oversized_string_field_with_visible_marker():
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    big_doc = "x" * 100_000
    t.emit("tool_call", name="fetch_doc", ok=True, result=big_doc)
    t.emit("agent_end", status="ok")

    import json

    tool_call_record = json.loads(t.to_jsonl().splitlines()[1])
    assert len(tool_call_record["result"]) < len(big_doc)
    assert "...[truncated" in tool_call_record["result"]
    assert "chars]" in tool_call_record["result"]


def test_emit_guarantees_line_length_regardless_of_field_size():
    # Property-style: for ANY field size, including several times over
    # the ceiling, the resulting to_jsonl() line for that event must
    # never exceed _MAX_TRACE_LINE_CHARS. This is the structural
    # guarantee the module docstring promises Task 2/3/4 authors — no
    # caller of emit() needs to size-check what it passes in.
    import arena.trace as trace_module

    field_sizes = [
        0,
        1,
        100,
        10_000,
        trace_module._MAX_EMITTED_LINE_CHARS - 1_000,  # just under budget
        trace_module._MAX_EMITTED_LINE_CHARS,  # right at budget
        trace_module._MAX_TRACE_LINE_CHARS,  # at validate()'s ceiling
        trace_module._MAX_TRACE_LINE_CHARS + 1,  # just over
        500_000,  # several x over
        2_000_000,  # far over
    ]
    for size in field_sizes:
        t = Trace(run_id="r1", seed=7)
        t.emit("agent_start", brief_id="b1")
        t.emit("tool_call", name="fetch_doc", ok=True, result="q" * size)
        t.emit("agent_end", status="ok")
        jsonl = t.to_jsonl()
        for line in jsonl.splitlines():
            assert len(line) <= trace_module._MAX_TRACE_LINE_CHARS, (
                f"field size {size}: emitted line of length {len(line)} "
                f"exceeds the validate() ceiling of "
                f"{trace_module._MAX_TRACE_LINE_CHARS}"
            )
        ok, reason = Trace.validate(jsonl)
        assert ok is True and reason == "", (
            f"field size {size}: expected clean validate(), got "
            f"({ok!r}, {reason!r})"
        )


def test_emit_truncates_oversized_non_string_field():
    # A non-string field (e.g. a big list of citation indices) can't be
    # sliced like a string, but must still be handled without raising
    # and without blowing the budget — replaced with a short marker
    # describing what was elided.
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("tool_call", name="fetch_doc", ok=True, citations=list(range(200_000)))
    t.emit("agent_end", status="ok")
    jsonl = t.to_jsonl()
    for line in jsonl.splitlines():
        assert len(line) <= 100_000
    ok, reason = Trace.validate(jsonl)
    assert ok is True and reason == ""

    import json

    tool_call_record = json.loads(jsonl.splitlines()[1])
    assert isinstance(tool_call_record["citations"], str)
    assert "truncated" in tool_call_record["citations"]


def test_emit_truncation_is_deterministic():
    # Truncation must not break the determinism guarantee: same seed,
    # same emit sequence (including an oversized field) => byte-identical
    # to_jsonl() output.
    def build():
        t = Trace(run_id="r1", seed=42)
        t.emit("agent_start", brief_id="b1")
        t.emit("tool_call", name="fetch_doc", ok=True, result="d" * 250_000)
        t.emit("agent_end", status="ok")
        return t.to_jsonl()

    a = build()
    b = build()
    assert a == b


def test_emit_small_fields_are_never_truncated():
    # Regression guard: ordinary small fields must pass through
    # untouched — truncation must only kick in when actually needed.
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("tool_call", name="search", ok=True, query="capital of Vietnam")
    t.emit("agent_end", status="ok")
    events = t.to_jsonl().splitlines()

    import json

    tool_call_record = json.loads(events[1])
    assert tool_call_record["query"] == "capital of Vietnam"
    assert "truncated" not in tool_call_record["query"]


def test_validate_recursion_error_from_json_loads_does_not_raise(monkeypatch):
    # Directly exercises the broad `except Exception` around json.loads
    # (Defense 2) in isolation from the length ceiling (Defense 1): even
    # a short, well-under-the-ceiling line must not propagate a
    # RecursionError (or any other exception type) raised by the parser.
    import arena.trace as trace_module

    def boom(_s):
        raise RecursionError("Stack overflow (simulated for test)")

    monkeypatch.setattr(trace_module.json, "loads", boom)
    ok, reason = Trace.validate('{"seq":0,"event":"agent_start"}')
    assert ok is False
    assert "malformed json" in reason
    assert "RecursionError" in reason


# validate() must never raise, full stop — sweep every pathological input
# above (and a few more) through validate() and assert each comes back as a
# plain 2-tuple, never an exception.


PATHOLOGICAL_INPUTS = [
    "",
    "   ",
    "\n\n\n",
    "null",
    "[]",
    "true",
    "42",
    '"just a string"',
    "{not valid json,,,",
    '{"seq": 0}',  # missing event
    '{"event": "agent_start"}',  # missing seq
    '{"seq": null, "event": "agent_start"}',
    '{"seq": 0, "event": null}',
    '{"seq": 0, "event": "agent_start"}\n{"seq": 0, "event": "agent_end"}',
    '{"seq": 0, "event": "agent_start"}\nnot json at all',
    '{"seq": [1,2], "event": "agent_start"}',
    '{"seq": {}, "event": "agent_start"}',
    '{"seq": true, "event": "agent_start"}',
    _ARRAY_BOMB_LINE,
    _OBJECT_BOMB_LINE,
]


def test_validate_never_raises_on_pathological_input():
    for bad_input in PATHOLOGICAL_INPUTS:
        result = Trace.validate(bad_input)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"validate() must return a 2-tuple, got {result!r} for "
            f"input {bad_input!r}"
        )
        ok, reason = result
        assert isinstance(ok, bool), (
            f"ok must be a bool for input {bad_input!r}, got {ok!r}"
        )
        assert isinstance(reason, str), (
            f"reason must be a str for input {bad_input!r}, got {reason!r}"
        )


# ---------------------------------------------------------------------------
# Performance probe: validate() must stay fast at competition scale.
# ---------------------------------------------------------------------------


def test_validate_performance_100k_events():
    t = Trace(run_id="perf", seed=1)
    t.emit("agent_start", brief_id="b1")
    for i in range(100_000):
        t.emit("tool_call", name=f"tool{i % 7}", ok=True)
    t.emit("agent_end", status="ok")
    jsonl = t.to_jsonl()

    start = time.perf_counter()
    ok, reason = Trace.validate(jsonl)
    elapsed = time.perf_counter() - start

    assert ok is True and reason == ""
    # Generous ceiling — the reviewer measured ~0.14s on this box; fail
    # loudly if validate() regresses to something quadratic.
    assert elapsed < 5.0, f"validate() took {elapsed:.3f}s for 100k events"


# ---------------------------------------------------------------------------
# Fix round 4 — the structural guarantee, for real this time.
#
# Round 3's budget fitter closed the "one oversized value" hole but left a
# worse one: its last-resort fallback kept only the RESERVED fields
# (seq/event/run_id/seed) and threw away `name`/`ok`/`prompt_tokens`/
# `completion_tokens` — the very fields validate() requires. So a caller who
# attached several ordinary metadata fields to a tool_call could have emit()
# succeed and then fail the gate with the MISLEADING reason "tool_call
# missing field: name", despite having passed `name`.
#
# Fix: `_REQUIRED_FIELDS_BY_EVENT` is now the single source of truth read by
# BOTH validate() (rules 2/3) and the budget fitter (which fields it must
# never drop), and the fitter bounds field NAME length and field COUNT, not
# just value size. The tests below cover every shape that broke it, plus a
# property-style sweep over the whole space — the test that should have
# existed from the start.
# ---------------------------------------------------------------------------


REQUIRED_FIELD_VALUES = {
    "tool_call": {"name": "fetch_doc", "ok": True},
    "model_call": {"prompt_tokens": 12, "completion_tokens": 34},
}


def _trace_with_tool_call(**fields):
    """agent_start / tool_call(**fields) / agent_end, ready to validate."""
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("tool_call", name="fetch_doc", ok=True, **fields)
    t.emit("agent_end", status="ok")
    return t


def _assert_gate_passes(t, label):
    import arena.trace as trace_module

    jsonl = t.to_jsonl()
    for line in jsonl.splitlines():
        assert len(line) <= trace_module._MAX_TRACE_LINE_CHARS, (
            f"{label}: emitted line of length {len(line)} exceeds the "
            f"validate() ceiling of {trace_module._MAX_TRACE_LINE_CHARS}"
        )
    ok, reason = Trace.validate(jsonl)
    assert ok is True and reason == "", (
        f"{label}: emit() produced a trace the gate rejects: "
        f"({ok!r}, {reason!r})"
    )
    return jsonl


def test_emit_five_thousand_small_fields_still_passes_the_gate():
    # The realistic break: a Task 3/4 author attaching a pile of metadata
    # (timings, retry counts, headers, debug notes) to one tool_call. No
    # single field is large; the COUNT is what blew the budget.
    fields = {f"meta_{i}": "x" * 30 for i in range(5_000)}
    _assert_gate_passes(_trace_with_tool_call(**fields), "5000 small fields")


def test_emit_twenty_thousand_tiny_fields_still_passes_the_gate():
    fields = {f"f{i}": "x" for i in range(20_000)}
    _assert_gate_passes(_trace_with_tool_call(**fields), "20000 tiny fields")


def test_emit_huge_field_name_still_passes_the_gate():
    # A 200,000-char field NAME with a short value: value-only truncation
    # (round 3) could never fix this, because the name is the payload.
    jsonl = _assert_gate_passes(
        _trace_with_tool_call(**{"n" * 200_000: "short"}), "huge field name"
    )
    record = json.loads(jsonl.splitlines()[1])
    assert all(len(key) < 1_000 for key in record), (
        "an over-long field name must be shortened, not carried verbatim"
    )


def test_emit_huge_field_name_and_huge_value_still_passes_the_gate():
    _assert_gate_passes(
        _trace_with_tool_call(**{"n" * 200_000: "v" * 200_000}),
        "huge name + huge value",
    )


def test_emit_several_oversized_fields_at_once_still_passes_the_gate():
    fields = {f"big_{i}": "z" * 200_000 for i in range(5)}
    _assert_gate_passes(_trace_with_tool_call(**fields), "5 oversized fields")


def test_emit_preserves_required_fields_under_budget_pressure():
    # The exact regression: the fields validate() requires must survive
    # even when everything else has to be dropped.
    for event, required in REQUIRED_FIELD_VALUES.items():
        t = Trace(run_id="r1", seed=7)
        t.emit("agent_start", brief_id="b1")
        t.emit(event, **required, **{f"m{i}": "x" * 30 for i in range(5_000)})
        t.emit("agent_end", status="ok")
        jsonl = _assert_gate_passes(t, f"{event} under pressure")
        record = json.loads(jsonl.splitlines()[1])
        for field in required:
            assert field in record, (
                f"{event}: required field {field!r} was dropped by the "
                f"budget fitter — this is the round-3 bug"
            )
        assert record.get("fields_truncated") is True
        assert record.get("fields_dropped", 0) > 0


def test_emit_keeps_required_field_keys_even_when_their_values_are_huge():
    # Presence, not byte-exactness, is what the gate checks: an enormous
    # `name`/`ok` may be shortened, but the KEYS must survive.
    t = Trace(run_id="R" * 300_000, seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("tool_call", name="N" * 200_000, ok=["huge"] * 100_000,
           blob="b" * 200_000)
    t.emit("agent_end", status="ok")
    jsonl = _assert_gate_passes(t, "huge protected values")
    record = json.loads(jsonl.splitlines()[1])
    assert "name" in record and "ok" in record
    assert record["run_id"]  # reserved fields survive too, just shortened


def test_emit_does_not_invent_missing_required_fields_and_reason_is_true():
    # If the CALLER never passed a required field, the gate must still
    # fail — but with the TRUE reason. emit() must not paper over it, and
    # must not report a field the caller actually did pass.
    t = Trace(run_id="r1", seed=7)
    t.emit("agent_start", brief_id="b1")
    t.emit("tool_call", ok=True, **{f"m{i}": "x" * 30 for i in range(5_000)})
    t.emit("agent_end", status="ok")
    ok, reason = Trace.validate(t.to_jsonl())
    assert ok is False
    assert "name" in reason and "tool_call" in reason
    # `ok` WAS passed, so it must not be the field blamed.
    assert "missing field: ok" not in reason


def test_required_fields_are_a_single_source_of_truth():
    # Anti-drift: validate()'s rules 2/3 and emit()'s "never drop these"
    # set must both be derived from _REQUIRED_FIELDS_BY_EVENT. Adding a
    # rule to that mapping without teaching the fitter about it is the
    # exact class of bug this test exists to make impossible.
    import arena.trace as trace_module

    mapping = trace_module._REQUIRED_FIELDS_BY_EVENT
    assert mapping["tool_call"] == ("name", "ok")
    assert mapping["model_call"] == ("prompt_tokens", "completion_tokens")

    for event, fields in mapping.items():
        for omitted in fields:
            # (a) validate() enforces every declared field...
            present = {
                k: v
                for k, v in REQUIRED_FIELD_VALUES[event].items()
                if k != omitted
            }
            t = Trace(run_id="r1", seed=7)
            t.emit("agent_start", brief_id="b1")
            t.emit(event, **present)
            t.emit("agent_end", status="ok")
            ok, reason = Trace.validate(t.to_jsonl())
            assert ok is False and omitted in reason and event in reason

            # (b) ...and the budget fitter protects every declared field.
            record = {
                "seq": 1,
                "event": event,
                "run_id": "r1",
                "seed": 7,
                **REQUIRED_FIELD_VALUES[event],
                **{f"pad{i}": "y" * 50 for i in range(5_000)},
            }
            fitted = trace_module._fit_record_to_budget(
                record, trace_module._MAX_EMITTED_LINE_CHARS
            )
            assert omitted in fitted, (
                f"{event}: fitter dropped declared required field {omitted!r}"
            )


def test_emit_gate_guarantee_property():
    # THE property test: across a spread of field counts, field-name
    # lengths, value sizes and value types — including pathological
    # combinations — emit() -> to_jsonl() -> validate() must ALWAYS be
    # (True, ""). Each previous fix round repaired the one reported input
    # and missed the next shape; this sweeps the space instead.
    import arena.trace as trace_module

    budget = trace_module._MAX_EMITTED_LINE_CHARS
    ceiling = trace_module._MAX_TRACE_LINE_CHARS
    field_counts = [0, 1, 2, 100, 501, 5_000, 20_000]
    name_lengths = [1, 8, 200, 201, 5_000, 200_000]
    value_makers = {
        "empty": lambda n: "",
        "small": lambda n: "x" * 30,
        "unicode": lambda n: "é\U0001F600\x00" * 10,
        "at_budget": lambda n: "v" * budget,
        "over_ceiling": lambda n: "w" * (ceiling + 1),
        "list": lambda n: list(range(5_000)),
        "dict": lambda n: {"nested": "d" * 50_000},
        "none": lambda n: None,
    }

    for count in field_counts:
        for name_len in name_lengths:
            for kind, make in value_makers.items():
                # Skip only the combinations that would be gigabytes of
                # test data, not any that are merely hostile.
                if count * (name_len + 1) > 4_000_000:
                    continue
                if count > 100 and kind in ("at_budget", "over_ceiling",
                                            "list", "dict"):
                    continue
                fields = {
                    ("k" * name_len) + str(i): make(i) for i in range(count)
                }
                for event in ("tool_call", "model_call", "layer"):
                    t = Trace(run_id="r1", seed=7)
                    t.emit("agent_start", brief_id="b1")
                    t.emit(event, **REQUIRED_FIELD_VALUES.get(event, {}),
                           **fields)
                    t.emit("agent_end", status="ok")
                    _assert_gate_passes(
                        t,
                        f"event={event} count={count} name_len={name_len} "
                        f"value={kind}",
                    )


def test_emit_reshaping_is_deterministic_for_pathological_input():
    # Determinism must survive the new name-shortening and field-dropping
    # paths too, not just value truncation.
    def build():
        t = Trace(run_id="r1", seed=42)
        t.emit("agent_start", brief_id="b1")
        t.emit(
            "tool_call",
            name="fetch_doc",
            ok=True,
            **{("n" * 5_000) + str(i): "z" * 1_000 for i in range(2_000)},
        )
        t.emit("agent_end", status="ok")
        return t.to_jsonl()

    assert build() == build()
