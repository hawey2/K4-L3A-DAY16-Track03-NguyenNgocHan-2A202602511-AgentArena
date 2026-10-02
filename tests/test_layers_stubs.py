"""The five layers you own — contract, safety net, and two lessons.

This is the suite to run while you build. Every test here holds BOTH on a
fresh checkout (all five stubs are no-ops) AND once you have implemented
them, because none of it asserts what a layer should compute. It asserts
the things that are true of a layer that WORKS AT ALL:

1. the class exists under the name the harness, the scripts and the deck
   all use, and it is a `Middleware`;
2. installing it does not break the run — no exception, trace gate still
   green, a report still submitted;
3. the two edits that look harmless and are not (rewriting claim text,
   returning something that is not a dict).

What it deliberately does NOT do is score you. `python3
scripts/run_practice.py` does that, and `scripts/leaderboard.py` turns it
into the number that matters — the GAP against your own no-layers
baseline.

WHY THERE IS NO "EACH LAYER ALONE MUST GAIN POINTS" TEST
========================================================
Because it would be wrong. Measured on the build tree: `retry` ALONE is
worth -0.65 and is positive on only 6 of 40 seeds, since without
`citation_checker` the evidence it recovers gets misattributed anyway. A
layer is judged LEAVE-ONE-OUT — remove it from the complete stack and see
whether the score falls — never in isolation. Keep that in mind before
you conclude that a layer you just wrote "does nothing".
"""

from __future__ import annotations

import pytest

from arena.scorer import score_run
from arena.trace import Trace

from harness.middleware import Middleware

from tests.fixtures_briefs import (
    BRIEF_SLA,
    CORPUS,
    SEEDS,
    STACK_ORDER,
    TRAP_BRIEFS,
    run,
)

SEED = SEEDS[0]


def _layer_classes() -> dict:
    from harness.layers.budget_policy import BudgetPolicy
    from harness.layers.citation_checker import CitationChecker
    from harness.layers.critic import Critic
    from harness.layers.injection_guard import InjectionGuard
    from harness.layers.retry import Retry

    return {
        "injection_guard": InjectionGuard,
        "critic": Critic,
        "citation_checker": CitationChecker,
        "budget_policy": BudgetPolicy,
        "retry": Retry,
    }


def student_stack(names=None) -> list:
    """The five layers, in the documented installation order."""
    classes = _layer_classes()
    wanted = set(classes) if names is None else set(names)
    return [classes[name]() for name in STACK_ORDER if name in wanted]


# ---------------------------------------------------------------------------
# 1. The contract
# ---------------------------------------------------------------------------


def test_all_five_layers_exist_under_their_documented_names():
    classes = _layer_classes()
    assert set(classes) == set(STACK_ORDER)
    for name, cls in classes.items():
        assert issubclass(cls, Middleware), name
        assert cls().label == name, (name, cls().label)


def test_the_installation_order_is_the_one_the_scripts_use():
    """`scripts/run_practice.py` builds the stack from the same tuple. If
    these ever disagree, the order you reason about in
    `harness/middleware.py` is not the order you are graded in."""
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from run_practice import STACK_ORDER as SCRIPT_ORDER

    assert tuple(SCRIPT_ORDER) == tuple(STACK_ORDER)


# ---------------------------------------------------------------------------
# 2. The safety net — your layers must not break the run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", STACK_ORDER)
def test_each_layer_alone_leaves_a_runnable_agent(name):
    report, jsonl = run(BRIEF_SLA, SEED, student_stack([name]))
    assert isinstance(report, dict), name
    ok, reason = Trace.validate(jsonl)
    assert ok, (name, reason)


@pytest.mark.parametrize("brief", TRAP_BRIEFS, ids=lambda b: b["brief_id"])
def test_the_whole_stack_keeps_the_trace_gate_green_on_every_trap(brief):
    """The gate is PASS/FAIL and a failure is a zero on that brief,
    whatever the report says. It passes for free while you keep using the
    harness — this test is here to tell you the moment you stop."""
    report, jsonl = run(brief, SEED, student_stack())
    ok, reason = Trace.validate(jsonl)
    assert ok, (brief["brief_id"], reason)
    score = score_run(brief, report, trace_jsonl=jsonl, corpus=CORPUS)
    assert score.gate_passed, (brief["brief_id"], score.gate_reason)


def test_the_whole_stack_still_submits_a_report_on_every_trap():
    for offset, brief in enumerate(TRAP_BRIEFS):
        report, _ = run(brief, SEED + offset, student_stack())
        assert isinstance(report, dict), brief["brief_id"]
        assert set(report) >= {"answer", "claims", "citations", "abstain"} or report == {}, (
            brief["brief_id"],
            sorted(report),
        )


def test_a_layer_that_raises_takes_the_run_down_with_it():
    """Hooks are NOT sandboxed, on purpose: a layer that silently swallowed
    its own bugs would be worse during practice than one that fails loudly.
    In the scored round the runner catches it, records the error and the
    entry scores what an empty run scores."""

    class Exploding(Middleware):
        name = "exploding"

        def after_agent(self, ctx, report):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run(BRIEF_SLA, SEED, [Exploding()])


def test_after_agent_must_return_a_dict():
    """The single most common way a first `critic` breaks a run: an early
    `return` on a branch that meant to fall through."""

    class Forgetful(Middleware):
        name = "forgetful"

        def after_agent(self, ctx, report):
            report.get("claims")  # ...and forgets to return it

    with pytest.raises(TypeError, match="after_agent must return a dict"):
        run(BRIEF_SLA, SEED, [Forgetful()])


# ---------------------------------------------------------------------------
# 3. The two lessons that cost the most points
# ---------------------------------------------------------------------------


def test_rewriting_claim_text_destroys_the_claim():
    """READ THIS BEFORE YOU WRITE `critic` OR `citation_checker`.

    A claim is credited only if it is (a) text the MODEL produced and (b) a
    verbatim quotation of ONE LINE of the document it cites. Adding a full
    stop, straightening a quote mark, normalising whitespace or repairing a
    truncated sentence breaks BOTH tests at once. Measured on the complete
    reference stack: a trailing period costs 47.16 points and lands the run
    on the abstain-everything floor. Trimming to a SUBSTRING is legal — a
    substring of a line is still a quotation of that line.
    """

    class AddsAPeriod(Middleware):
        name = "adds-a-period"

        def after_agent(self, ctx, report):
            claims = report.get("claims")
            if isinstance(claims, list):
                report["claims"] = [
                    {**c, "text": str(c.get("text", "")) + "."} for c in claims
                ]
            return report

    class TrimsInstead(Middleware):
        name = "trims-instead"

        def after_agent(self, ctx, report):
            claims = report.get("claims")
            if isinstance(claims, list):
                report["claims"] = [
                    {**c, "text": str(c.get("text", ""))[:60]} for c in claims
                ]
            return report

    def total(layers):
        report, jsonl = run(BRIEF_SLA, SEED, layers)
        return score_run(BRIEF_SLA, report, trace_jsonl=jsonl, corpus=CORPUS)

    plain = total(None)
    edited = total([AddsAPeriod()])
    trimmed = total([TrimsInstead()])

    assert plain.grounding > 0.0, "the fixture must earn some grounding to begin with"
    assert edited.grounding < plain.grounding, (plain.grounding, edited.grounding)
    # A substring keeps its provenance and its support...
    assert trimmed.grounding >= edited.grounding, (trimmed.grounding, edited.grounding)


def test_a_claim_the_model_never_wrote_is_not_yours_to_add():
    """The other half of the same rule, and the reason a "helpful" layer
    that pastes the best line of a document into the report scores WORSE
    than one that deletes claims: the scorer credits only text that appears
    in some `model_call`'s parsed FINAL, cited to a document the run
    actually retrieved. Inventing evidence is priced like fabricating it.
    """
    plain_report, plain_jsonl = run(BRIEF_SLA, SEED, None)

    # A document this run never touched, so the pasted line cannot be
    # anything the model said. Chosen from the trace rather than hard-coded:
    # the mock's retrieval changes with the seed.
    seen = {doc.doc_id for doc in CORPUS.docs if doc.doc_id in plain_jsonl}
    victim = next(doc for doc in CORPUS.docs if doc.doc_id not in seen)
    line = max(victim.body.splitlines(), key=len).strip()

    class PastesTheCorpus(Middleware):
        name = "pastes-the-corpus"

        def after_agent(self, ctx, report):
            claims = list(report.get("claims") or [])
            claims.append({"text": line, "doc_id": victim.doc_id})
            report["claims"] = claims
            return report

    pasted_report, pasted_jsonl = run(BRIEF_SLA, SEED, [PastesTheCorpus()])
    assert pasted_report["claims"][-1]["text"] == line

    plain = score_run(BRIEF_SLA, plain_report, trace_jsonl=plain_jsonl, corpus=CORPUS)
    pasted = score_run(BRIEF_SLA, pasted_report, trace_jsonl=pasted_jsonl, corpus=CORPUS)

    last = len(pasted_report["claims"]) - 1
    verdict = next(
        c["verdict"] for c in pasted.detail["grounding"]["claims"] if c["index"] == last
    )
    assert verdict != "SUPPORTED", verdict
    assert pasted.total <= plain.total, (plain.total, pasted.total)
