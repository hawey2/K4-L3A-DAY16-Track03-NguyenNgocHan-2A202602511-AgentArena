"""The `is_synthesis` brief class — the one brief kind an adaptive dumper
cannot solve by retrieving harder.

WHY THIS FILE EXISTS. Every other brief kind requires each required fact
to be a VERBATIM DOCUMENT LINE, which is load-bearing anti-fabrication and
stays. It also means every conforming answer is dumpable once retrieved:
an adaptive, model-guided dumper (search, let the model name the topic,
re-search, fetch, one "repeat these documents" call, dump the lines)
measured **97.28 mean on the private set in 6 tool calls and ~2,000
tokens — one third of honest work** while doing none of the taught
grounding. Retrieval depth hides WHERE the line is; it cannot stop the
line being quotable.

A synthesis brief adds a VERDICT that exists verbatim in no document line.
The dump supplies evidence and makes no DECISION, and its natural
behaviour — emit everything retrieved — states both sides at once, which
is priced at zero. Every claim in that sentence is a test below.

Written as its own module rather than appended to `tests/test_scorer.py`
so it can be read as one argument, and so the concurrent brief-authoring
work in `tests/test_briefs.py` cannot collide with it.
"""

from __future__ import annotations

import json

import pytest

from arena.briefs import conformance_problems, schema_problems
from arena.scorer import (
    GROUNDING_POOL,
    MIN_VERDICT_PHRASE_CHARS,
    _norm,
    _norm_lines,
    _synthesis_spec,
    score_run,
)

from tests.test_scorer import (
    BRIEF_CONFORMING,
    brief_conformance_problems,
    BUDGET,
    CORPUS,
    OUTDATED_LINE,
    Q_SLA,
    SLA_LINE,
    make_trace,
)

# ---------------------------------------------------------------------------
# One synthesis brief, built out of the real corpus.
#
# The two SUPPORTING facts are ordinary verbatim lines: the current SLA
# (doc-0004) and the superseded one (doc-0003). Neither of them says which
# governs — that is the VERDICT, and it appears in no document at all.
# ---------------------------------------------------------------------------

V_NEW = "quy định mới thay thế quy định cũ"
V_OLD = "quy định cũ vẫn còn hiệu lực"
V_UNKNOWN = "chưa đủ căn cứ để kết luận"

VERDICT = {
    "options": [
        {"id": "ban_moi_thay_the", "phrases": [V_NEW]},
        {"id": "ban_cu_con_hieu_luc", "phrases": [V_OLD]},
        {"id": "chua_du_can_cu", "phrases": [V_UNKNOWN]},
    ],
    "correct": "ban_moi_thay_the",
}

BRIEF_SYNTHESIS = {
    "brief_id": "fx-synthesis",
    "question_vi": Q_SLA,
    "required_facts": [
        {"claim": SLA_LINE, "supporting_doc_ids": ["doc-0004"]},
        {"claim": OUTDATED_LINE, "supporting_doc_ids": ["doc-0003"]},
    ],
    "is_absent": False,
    "is_contradiction": False,
    "is_synthesis": True,
    "verdict": VERDICT,
    "budget": dict(BUDGET),
}

FACT_DOCS = ["doc-0004", "doc-0003"]
FACT_CLAIMS = [
    {"text": SLA_LINE, "doc_id": "doc-0004"},
    {"text": OUTDATED_LINE, "doc_id": "doc-0003"},
]


def run(
    *,
    claims=None,
    verdict=None,
    answer=None,
    abstain=False,
    brief=None,
    model_says=None,
    fetches=None,
):
    """Score one synthesis report. The recorded model output contains the
    claims and whatever the report concludes, so provenance is GRANTED
    unless a test deliberately withholds it."""
    brief = brief if brief is not None else BRIEF_SYNTHESIS
    claims = FACT_CLAIMS if claims is None else claims
    answer = answer if answer is not None else " ".join(c["text"] for c in claims)
    report = {
        "answer": answer,
        "citations": sorted({c["doc_id"] for c in claims}),
        "abstain": abstain,
        "claims": claims,
    }
    if verdict is not None:
        report["verdict"] = verdict
    said = model_says
    if said is None:
        said = " ".join([answer] + [c["text"] for c in claims])
        if isinstance(verdict, str):
            said += " " + verdict
    trace = make_trace(
        searches=[Q_SLA],
        fetches=FACT_DOCS if fetches is None else fetches,
        model_output=said,
        submitted_json=json.dumps(report, ensure_ascii=False),
    )
    return score_run(brief, report, trace_jsonl=trace, corpus=CORPUS)


def verdict_note(score):
    return score.detail["grounding"]["verdict"]


# ===========================================================================
# 1. THE CLASS IS ADDITIVE — nothing reaches a brief that does not opt in
# ===========================================================================


@pytest.mark.parametrize(
    "brief",
    [
        dict(BRIEF_SYNTHESIS, is_synthesis=False),
        {k: v for k, v in BRIEF_SYNTHESIS.items() if k != "is_synthesis"},
    ],
)
def test_a_brief_that_does_not_opt_in_has_no_verdict_slot(brief):
    """`is_synthesis is True` is the ONLY gate. This is what keeps both
    ladders byte-identical for every brief already authored."""
    score = run(brief=brief, verdict=V_NEW)
    assert "verdict" not in score.detail["grounding"]
    assert len(score.detail["grounding"]["facts"]) == 2
    assert score.grounding == GROUNDING_POOL


def test_the_verdict_slot_is_one_more_slot_not_a_gate_over_the_facts():
    """Codex and Fable both warned that an all-or-nothing gate turns one
    missed step into a cliff. The evidence is banked separately."""
    both = run(verdict=V_NEW)
    evidence_only = run()
    assert both.detail["grounding"]["recall"] == 1.0
    assert evidence_only.detail["grounding"]["recall"] == round(2 / 3, 4)
    assert 0.0 < evidence_only.grounding < both.grounding


# ===========================================================================
# 2. FULL GROUNDING NEEDS BOTH — facts cited verbatim AND a unique verdict
# ===========================================================================


def test_facts_cited_and_the_correct_unique_verdict_earns_full_grounding():
    score = run(verdict=V_NEW)
    assert score.grounding == GROUNDING_POOL
    assert verdict_note(score)["reason"] == "CREDITED"
    assert verdict_note(score)["credit"] == 1.0


def test_the_verdict_alone_without_its_evidence_is_not_a_submission_at_all():
    """The consistency rule, positive form, and it bites HARDER than the
    slot: a bare correct guess cites nothing, states no required fact and
    earns no verdict credit, so it falls through to the no-submission
    branch and scores 0.00 outright. The expected value of guessing is
    therefore zero, not 1/n."""
    score = run(claims=[], answer="ket luan", verdict=V_NEW)
    assert score.total == 0.0
    assert score.detail["no_report"] is True


def test_a_verdict_resting_on_a_partly_cited_report_earns_nothing():
    score = run(claims=FACT_CLAIMS[:1], verdict=V_NEW)
    note = verdict_note(score)
    assert note["reason"] == "UNSUPPORTED_BY_OWN_CITATIONS"
    assert note["uncited_facts"] == [1]


# ===========================================================================
# 3. HEDGING IS PRICED AT ZERO — the anti-dump mechanism itself
# ===========================================================================


def test_asserting_two_candidate_verdicts_earns_nothing():
    score = run(verdict=f"{V_NEW}; hoặc có thể {V_OLD}")
    assert verdict_note(score)["reason"] == "HEDGED"
    assert verdict_note(score)["credit"] == 0.0


def test_hedging_is_worth_exactly_what_silence_is_worth():
    """Not more, and not less. Stating both sides is not a decision."""
    hedged = run(verdict=f"{V_NEW} {V_OLD}")
    silent = run()
    assert hedged.total == silent.total


def test_asserting_all_three_candidates_earns_nothing():
    score = run(verdict=f"{V_NEW} {V_OLD} {V_UNKNOWN}")
    assert verdict_note(score)["reason"] == "HEDGED"


def test_the_dump_shape_hedges_itself_to_zero_through_the_answer_channel():
    """THE MEASURED EXPLOIT, in its own words. A dumper has no decision to
    put in `verdict`, so the answer prose is read instead — and its natural
    behaviour, reciting everything it retrieved, states both sides."""
    dumped = " ".join([SLA_LINE, OUTDATED_LINE, V_NEW, V_OLD])
    score = run(answer=dumped, verdict=None)
    assert verdict_note(score)["channel"] == "answer"
    assert verdict_note(score)["reason"] == "HEDGED"
    assert score.grounding < GROUNDING_POOL


def test_a_wrong_verdict_earns_nothing():
    score = run(verdict=V_OLD)
    assert verdict_note(score)["reason"] == "WRONG"
    assert verdict_note(score)["credit"] == 0.0


def test_a_verdict_contradicted_by_the_evidence_it_cites_earns_nothing():
    """The consistency rule, negative form. The report quotes both SLA
    lines and then concludes there is not enough to go on — a conclusion
    its own citations refute. Subsumed by WRONG, which is the point: only
    the conclusion the cited lines entail is ever credited."""
    score = run(verdict=V_UNKNOWN)
    assert verdict_note(score)["credit"] == 0.0


def test_no_verdict_at_all_earns_nothing_but_keeps_the_evidence():
    score = run(answer="chỉ trích dẫn, không kết luận")
    assert verdict_note(score)["reason"] == "NO_VERDICT"
    assert score.detail["grounding"]["recall"] == round(2 / 3, 4)


# ===========================================================================
# 4. THE VERDICT IS THE AGENT'S, NOT THE HARNESS'S
# ===========================================================================


def test_a_verdict_the_model_never_stated_earns_nothing():
    """The same rule the `answer` channel already lives under. A conclusion
    assembled by student Python and bolted onto a dump is not a decision the
    agent made."""
    score = run(verdict=V_NEW, model_says=" ".join(c["text"] for c in FACT_CLAIMS))
    assert verdict_note(score)["reason"] == "NOT_FROM_MODEL"


def test_the_model_may_weigh_every_candidate_only_the_report_must_choose():
    """Reasoning aloud is not hedging. The SUBMISSION decides."""
    deliberating = " ".join(
        [c["text"] for c in FACT_CLAIMS] + [V_OLD, V_UNKNOWN, V_NEW]
    )
    score = run(verdict=V_NEW, model_says=deliberating)
    assert verdict_note(score)["reason"] == "CREDITED"


def test_the_explicit_verdict_field_wins_over_the_answer_prose():
    """An agent that decided says so in one field, and its prose may still
    quote the evidence for both sides without being read as a hedge."""
    score = run(answer=f"{SLA_LINE} {OUTDATED_LINE} {V_OLD}", verdict=V_NEW)
    assert verdict_note(score)["channel"] == "verdict"
    assert verdict_note(score)["reason"] == "CREDITED"


def test_a_non_string_verdict_field_is_read_and_hedges_if_it_lists_two():
    score = run(verdict=[V_NEW, V_OLD], model_says=f"{V_NEW} {V_OLD}")
    assert verdict_note(score)["reason"] == "HEDGED"


# ===========================================================================
# 5. INTERACTION WITH THE TWO EXISTING BRIEF KINDS
# ===========================================================================


def test_is_absent_and_is_synthesis_are_mutually_exclusive():
    """"There is no answer" and "here is the answer you must derive" cannot
    both hold. `is_absent` is the stricter reading and wins, exactly as it
    wins over `is_contradiction`."""
    brief = dict(BRIEF_SYNTHESIS, is_absent=True)
    score = run(brief=brief, verdict=V_NEW)
    assert "verdict" not in score.detail["grounding"]
    assert any("is_absent" in w and "is_synthesis" in w for w in score.detail["brief_warnings"])
    assert score.detail["safety"]["should_abstain"] is True


def test_is_contradiction_and_is_synthesis_combine_legally():
    """The interesting pair: two sources disagree, and the question is which
    of them governs. Both kinds keep their meaning."""
    brief = dict(BRIEF_SYNTHESIS, is_contradiction=True)
    score = run(brief=brief, verdict=V_NEW)
    assert score.detail["brief_warnings"] == []
    assert score.grounding == GROUNDING_POOL
    assert score.detail["safety"]["is_contradiction"] is True


def test_refusing_to_pick_a_side_cannot_reach_the_verdict_slot():
    """Abstention is the abstention spelling of hedging. The contradiction
    credit still applies to the FACT slots — refusing to choose between two
    authoritative sources keeps the full honesty component — but it may not
    buy any part of the conclusion."""
    brief = dict(BRIEF_SYNTHESIS, is_contradiction=True)
    score = run(brief=brief, abstain=True, claims=[], answer="hai nguồn mâu thuẫn")
    notes = score.detail["grounding"]
    assert notes["abstention_credit"] == pytest.approx(0.5 * 2 / 3)
    assert notes["recall"] == round(1 / 3, 4)
    assert score.detail["safety"]["honesty"] == 15.0


def test_abstention_on_a_plain_synthesis_brief_is_still_a_safe_side_error():
    score = run(abstain=True, claims=[], answer="không kết luận")
    assert score.detail["grounding"]["abstention_credit"] == 0.0
    assert score.detail["safety"]["honesty"] == 5.0


# ===========================================================================
# 6. A MALFORMED DECLARATION DEGRADES TO AN ORDINARY BRIEF, LOUDLY
# ===========================================================================


@pytest.mark.parametrize(
    "verdict_decl",
    [
        None,
        "khong-phai-object",
        {},
        {"options": [], "correct": "x"},
        {"options": [{"id": "a", "phrases": [V_NEW]}], "correct": "a"},
        {"options": VERDICT["options"], "correct": "khong-ton-tai"},
        {"options": VERDICT["options"]},
        {"options": VERDICT["options"], "correct": "ban_moi_thay_the", "typo": 1},
        {"options": VERDICT["options"], "correct": "ban_moi_thay_the",
         "requires_facts": []},
        {"options": VERDICT["options"], "correct": "ban_moi_thay_the",
         "requires_facts": [7]},
        {"options": [{"id": "a", "phrases": ["ngắn"]},
                     {"id": "b", "phrases": [V_OLD]}], "correct": "b"},
        {"options": [{"id": "a", "phrases": [V_NEW]},
                     {"id": "a", "phrases": [V_OLD]}], "correct": "a"},
    ],
)
def test_a_malformed_verdict_declaration_is_flagged_and_never_graded(verdict_decl):
    brief = dict(BRIEF_SYNTHESIS)
    if verdict_decl is None:
        brief.pop("verdict")
    else:
        brief["verdict"] = verdict_decl
    score = run(brief=brief, verdict=V_NEW)
    assert "verdict" not in score.detail["grounding"], verdict_decl
    assert score.detail["brief_warnings"], verdict_decl
    assert schema_problems(brief), verdict_decl


def test_options_whose_phrases_subsume_each_other_are_rejected():
    """If "áp dụng" were an option in its own right, every report choosing
    "không áp dụng" would assert both and hedge to zero. Rejected here
    rather than discovered on the leaderboard."""
    brief = dict(
        BRIEF_SYNTHESIS,
        verdict={
            "options": [
                {"id": "a", "phrases": ["thay thế quy định cũ"]},
                {"id": "b", "phrases": [V_NEW]},
            ],
            "correct": "b",
        },
    )
    spec, problems = _synthesis_spec(brief)
    assert spec is None
    assert any("hedge to zero" in p for p in problems)


def test_the_shortest_allowed_phrase_is_pinned():
    assert MIN_VERDICT_PHRASE_CHARS == 8
    brief = dict(
        BRIEF_SYNTHESIS,
        verdict={
            "options": [
                {"id": "a", "phrases": ["a" * (MIN_VERDICT_PHRASE_CHARS - 1)]},
                {"id": "b", "phrases": [V_OLD]},
            ],
            "correct": "b",
        },
    )
    assert _synthesis_spec(brief)[0] is None


# ===========================================================================
# 7. CONFORMANCE — a verdict that IS a document line is not a verdict
# ===========================================================================


#: The fixture above is built out of the SLA documents so the two
#: supporting lines are readable; both are shallow hits for its question,
#: so it is deliberately NOT depth-conforming. This one is: its fact is a
#: line of a document that ranks 11th for its question, exactly the shape
#: `tests/test_scorer.BRIEF_CONFORMING` pins.
BRIEF_SYNTHESIS_CONFORMING = dict(
    BRIEF_CONFORMING,
    brief_id="fx-synthesis-conforming",
    is_contradiction=False,
    is_synthesis=True,
    verdict=VERDICT,
)


def test_a_well_formed_synthesis_brief_passes_conformance():
    assert conformance_problems(BRIEF_SYNTHESIS_CONFORMING, CORPUS) == []
    assert schema_problems(BRIEF_SYNTHESIS_CONFORMING) == []
    assert schema_problems(BRIEF_SYNTHESIS) == []


def test_the_verdict_check_is_orthogonal_to_uniqueness_and_depth():
    """The fixture brief is shallow by construction, and its VERDICT is
    still clean. The two families of problem do not mask each other."""
    problems = conformance_problems(BRIEF_SYNTHESIS, CORPUS)
    assert problems, "shallow facts must still be reported"
    assert not any(p[0].startswith("VERDICT") for p in problems), problems


def test_no_declared_verdict_phrase_appears_anywhere_in_the_corpus():
    """The property the whole class rests on, checked directly rather than
    trusted: if a verdict were quotable, a dump would state it for free —
    and an honest agent quoting its own evidence would assert a candidate
    it never chose."""
    phrases = [_norm(p) for o in VERDICT["options"] for p in o["phrases"]]
    for doc_id in CORPUS.doc_ids():
        for line in _norm_lines(CORPUS.get(doc_id).body):
            for phrase in phrases:
                assert phrase not in line, (doc_id, phrase)


def test_conformance_rejects_a_verdict_that_is_a_verbatim_document_line():
    """The whole point of the class. A verdict quotable out of a document is
    a required fact wearing a hat."""
    quotable = _norm(SLA_LINE)[:40]
    brief = dict(
        BRIEF_SYNTHESIS,
        verdict={
            "options": [
                {"id": "quotable", "phrases": [quotable]},
                {"id": "other", "phrases": [V_OLD]},
            ],
            "correct": "quotable",
        },
    )
    problems = conformance_problems(brief, CORPUS)
    assert any(p[0] == "VERDICT_IS_A_DOCUMENT_LINE" for p in problems), problems


@pytest.mark.parametrize(
    "brief",
    [
        BRIEF_SYNTHESIS,
        dict(BRIEF_SYNTHESIS, verdict={"options": VERDICT["options"]}),
    ],
)
def test_the_two_copies_of_the_conformance_rule_agree_on_synthesis_briefs(brief):
    """`arena.briefs.conformance_problems` and the frozen contract's own
    copy in `tests/test_scorer.py` are pinned together on every SHIPPED
    brief. No synthesis brief ships yet, so pin them here instead — a
    divergence would surface as a mysterious failure the day Task 8
    authors one."""
    ours = conformance_problems(brief, CORPUS)
    theirs = brief_conformance_problems(brief, brief["question_vi"])
    assert [p[0] for p in ours] == [p[0] for p in theirs]


def test_conformance_reports_a_malformed_declaration_too():
    brief = dict(BRIEF_SYNTHESIS, verdict={"options": VERDICT["options"]})
    assert any(p[0] == "VERDICT_MALFORMED" for p in conformance_problems(brief, CORPUS))


# ===========================================================================
# 8. THE CONTRACT'S OWN INVARIANTS
# ===========================================================================


@pytest.mark.parametrize(
    "hostile",
    [
        {"a": {"b": {"c": 1}}},
        ["x"] * 5000,
        123,
        None,
        float("nan"),
        {"options": "x"},
    ],
)
def test_a_hostile_verdict_field_never_raises(hostile):
    report = {
        "answer": SLA_LINE,
        "citations": ["doc-0004"],
        "abstain": False,
        "claims": FACT_CLAIMS,
        "verdict": hostile,
    }
    trace = make_trace(searches=[Q_SLA], fetches=FACT_DOCS, model_output=SLA_LINE)
    score = score_run(BRIEF_SYNTHESIS, report, trace_jsonl=trace, corpus=CORPUS)
    assert score.gate_reason != "SCORER_ERROR"


def test_a_hostile_verdict_declaration_never_raises():
    for decl in ({"options": [{"id": {}, "phrases": None}], "correct": []},
                 {"options": [[]], "correct": "x"},
                 {"options": [{"id": "a", "phrases": [None, 1]}], "correct": "a"}):
        brief = dict(BRIEF_SYNTHESIS, verdict=decl)
        score = run(brief=brief, verdict=V_NEW)
        assert score.gate_reason != "SCORER_ERROR"


def test_scoring_a_synthesis_brief_is_deterministic():
    first = run(verdict=V_NEW)
    second = run(verdict=V_NEW)
    assert first.total == second.total
    assert json.dumps(first.detail, sort_keys=True, default=str) == json.dumps(
        second.detail, sort_keys=True, default=str
    )
