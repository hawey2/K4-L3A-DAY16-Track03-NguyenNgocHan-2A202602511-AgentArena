"""Tests for `arena/briefs.py` and the PRACTICE brief set.

WHAT THIS FILE IS, AND WHAT IT IS NOT
=====================================

The build tree's copy of this file is mostly about the SCORED set: it
proves every scored brief is unique, deep, enumerable, dispersed and
winnable, and it measures a few hundred thousand blind-dump attacks
against it. None of that can ship — the assertions ARE the scored briefs
— so what is left here is the half you can actually use:

* the brief SCHEMA, which is also the format you would write your own
  practice briefs in;
* the shape of the PUBLIC set: it spans every trap class, every brief
  baits the deliberately weak baseline with at least one trap, and no
  required fact hides inside the poisoned document;
* the two CONFORMING previews (`pub-08`, `pub-09`), which are authored
  the way the scored set is authored — and are therefore the two briefs
  in the bundle that tell you something true about the round.

THE ONE THING WORTH READING TWICE
=================================

Seven of the nine practice briefs are authored the EASY way: their
answer is the sentence `arena.model._lift_span` lifts out of a document
the question's own top-5 already returns. That is legal for practice and
worth nothing beyond it — "search once, quote the longest line of each
hit, cite each line to itself" scores well on those seven and near the
abstain floor on the scored set.

`pub-08` and `pub-09` are the counter-example, and they are in the bundle
on purpose. Their supporting document is NOT in the question's top
`MAX_SCORED_CLAIMS` hits, so quoting what retrieval returned cannot reach
it; you have to notice what is missing and ask a DIFFERENT question. The
notes ship a `refined_queries` entry showing one query that does reach it
— `test_the_conforming_previews_are_reachable_by_a_refined_query` runs it
— because discovering that mechanism during practice is the whole point
of shipping them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import-order guard
    sys.path.insert(0, str(REPO_ROOT))

from arena.briefs import (
    BRIEF_KEYS,
    BUDGET_KEYS,
    OPTIONAL_BRIEF_KEYS,
    PUBLIC_BRIEFS_PATH,
    BriefSetError,
    acceptance_problems,
    brief_by_id,
    conformance_problems,
    fact_is_verbatim,
    key_term_documents,
    load_brief_file,
    load_briefs,
    load_notes,
    load_public_briefs,
    schema_problems,
    trap_classes,
    trap_docs_in_top_k,
)
from arena.corpus import TRAP_CLASSES, Corpus
from arena.scorer import MAX_SCORED_CLAIMS, _norm

SEED = 42
CORPUS = Corpus.generate(seed=SEED)

PUBLIC = load_public_briefs()
PUBLIC_NOTES = load_notes(PUBLIC_BRIEFS_PATH)

#: The poisoned document. Never the home of a required fact: `_lift_span`
#: picks the injected INSTRUCTION as the document's quotable sentence, and
#: that sentence sits inside the block a working `injection_guard` strips
#: — so a correctly guarded agent structurally could not quote it.
INJECTION_DOC = "doc-0006"

#: The evidence-of-absence document. Contagious: it has been measured
#: landing in the top-5 of briefs that are NOT flagged `is_absent`, where
#: the model cheerfully fabricates the missing figure.
ABSENT_DOC = "doc-0007"


def _ids(briefs) -> list[str]:
    return [b["brief_id"] for b in briefs]


def _facts(brief):
    return brief["required_facts"]


def _pytest_ids(briefs):
    return {"argvalues": briefs, "ids": _ids(briefs)}


#: The briefs the notes claim are authored like the scored set.
CONFORMING = [b for b in PUBLIC if PUBLIC_NOTES.get(b["brief_id"], {}).get("conforming")]


# ===========================================================================
# 1. SCHEMA — the format, and the shapes that would grade the wrong thing
# ===========================================================================


@pytest.mark.parametrize("brief", **_pytest_ids(PUBLIC))
def test_every_shipped_brief_is_well_formed(brief):
    """Every REQUIRED key present, and nothing beyond the two OPTIONAL ones.

    `is_synthesis` / `verdict` are opt-in, so the assertion is a sandwich
    rather than an equality. A typo'd flag still fails it, which is the
    property that matters: the scorer ignores what it does not know, so an
    unknown key grades the wrong thing silently.
    """
    assert schema_problems(brief) == []
    assert set(BRIEF_KEYS) <= set(brief), sorted(set(BRIEF_KEYS) - set(brief))
    assert set(brief) <= set(BRIEF_KEYS) | set(OPTIONAL_BRIEF_KEYS), sorted(
        set(brief) - set(BRIEF_KEYS) - set(OPTIONAL_BRIEF_KEYS)
    )
    assert ("verdict" in brief) == (brief.get("is_synthesis") is True)
    assert set(brief["budget"]) == set(BUDGET_KEYS)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda b: {**b, "is_absent": True, "is_contradiction": True}, "mutually exclusive"),
        (lambda b: {**b, "required_facts": []}, "non-empty list"),
        (lambda b: {k: v for k, v in b.items() if k != "budget"}, "missing keys"),
        (lambda b: {**b, "budget": {**b["budget"], "max_tool_calls": 0}}, "positive number"),
        (lambda b: {**b, "extra": 1}, "unknown keys"),
        (lambda b: {**b, "is_absent": "yes"}, "must be a bool"),
    ],
)
def test_schema_rejects_the_shapes_that_grade_the_wrong_thing(mutate, expected):
    broken = mutate(dict(PUBLIC[0]))
    problems = schema_problems(broken)
    assert any(expected in p for p in problems), problems


def test_a_contradiction_brief_must_declare_both_sides():
    """One fact per conflicting document is what lets the scorer tell
    "surfaced the conflict" from "picked a side" — and picking a side is
    what the baseline does, by fusing them into one sentence neither
    document contains."""
    contradiction = [b for b in PUBLIC if b.get("is_contradiction")]
    assert contradiction, "the practice set must exercise the contradiction kind"
    for brief in contradiction:
        assert len(_facts(brief)) >= 2, brief["brief_id"]
        nominated = [tuple(f["supporting_doc_ids"]) for f in _facts(brief)]
        assert len(set(nominated)) == len(nominated), "one document per side"
        one_sided = {**brief, "required_facts": _facts(brief)[:1]}
        assert schema_problems(one_sided), "a one-sided contradiction brief must be rejected"


def test_the_two_kind_flags_are_never_both_set():
    for brief in PUBLIC:
        assert not (brief["is_absent"] and brief["is_contradiction"]), brief["brief_id"]


def test_a_missing_or_malformed_brief_file_is_loud(tmp_path):
    """A round that silently grades zero briefs is worse than one that
    refuses to start."""
    with pytest.raises(FileNotFoundError):
        load_briefs(tmp_path / "no-such-briefs.json")
    bad = tmp_path / "broken_briefs.json"
    for content in ("{}", '{"briefs": []}', "not json", '{"briefs": {}}'):
        bad.write_text(content, encoding="utf-8")
        with pytest.raises(BriefSetError):
            load_briefs(bad)


def test_the_loader_surface_is_the_one_the_scripts_use():
    whole = load_brief_file(PUBLIC_BRIEFS_PATH)
    assert whole["set"] == "public"
    assert whole["corpus_seed"] == SEED
    assert [b["brief_id"] for b in whole["briefs"]] == _ids(PUBLIC)
    assert brief_by_id(PUBLIC, PUBLIC[0]["brief_id"]) == PUBLIC[0]
    with pytest.raises(KeyError):
        brief_by_id(PUBLIC, "no-such-brief")


# ===========================================================================
# 2. WHAT EVERY FACT HAS TO BE
# ===========================================================================


@pytest.mark.parametrize("brief", **_pytest_ids(PUBLIC))
def test_every_required_fact_quotes_one_line_of_its_document(brief):
    """The scorer scopes support to a LINE. A brief demanding a splice
    across a title, a blank line and half a paragraph would be demanding
    something no agent can be credited for — and your `critic` must never
    "repair" a claim across that boundary either."""
    for fact in _facts(brief):
        assert fact_is_verbatim(fact, CORPUS), (brief["brief_id"], fact["claim"][:60])


@pytest.mark.parametrize("brief", **_pytest_ids(PUBLIC))
def test_key_terms_are_no_looser_than_the_claim(brief):
    """`key_terms` OVERRIDE the claim in the scorer's recall test, so they
    are a second surface a conformance check never sees. Any fact that
    pins them must pin them to its own document alone."""
    for fact in _facts(brief):
        if "key_terms" not in fact:
            continue
        assert key_term_documents(fact, CORPUS) == fact["supporting_doc_ids"], (
            brief["brief_id"],
            fact["key_terms"],
        )


@pytest.mark.parametrize("brief", **_pytest_ids(PUBLIC))
def test_no_required_fact_lives_in_filing_metadata_or_boilerplate(brief):
    """The two lines a line-shotgun attack hit at 100.00 were a
    pipe-delimited header and the "exceptions are approved by X" FAQ
    boilerplate. Facts are authored out of substantive prose, never out of
    either — which is also why quoting boilerplate earns you nothing."""
    for fact in _facts(brief):
        claim = fact["claim"]
        assert "|" not in claim, "filing metadata is not evidence"
        assert not claim.startswith("Đáp "), "FAQ exception boilerplate is not evidence"
        assert "phê duyệt bằng văn bản trước khi áp dụng" not in claim
        for doc_id in fact["supporting_doc_ids"]:
            doc = CORPUS.get(doc_id)
            assert _norm(claim) != _norm(doc.title), "a title is not evidence"


# ===========================================================================
# 3. THE PRACTICE SET IS LABELLED HONESTLY
#
# `conformance_problems` returns a NON-EMPTY list for a lift-span-authored
# brief, on purpose. The label has to be true rather than decorative, or a
# student mistakes a practice brief for a scored-round shape.
# ===========================================================================


def test_the_public_set_is_labelled_honestly():
    conforming = {b["brief_id"] for b in CONFORMING}
    assert conforming, "ship at least one CONFORMING practice brief"
    for brief in PUBLIC:
        is_conforming = conformance_problems(brief, CORPUS) == []
        if brief["brief_id"] in conforming:
            assert is_conforming, brief["brief_id"]
        elif not brief["is_absent"]:
            # `is_absent` briefs are exempt from the check by contract, so
            # they come back clean without being authored the hard way.
            assert not is_conforming, (
                f"{brief['brief_id']} conforms but is not labelled as such"
            )


@pytest.mark.parametrize("brief", **_pytest_ids(CONFORMING))
def test_the_conforming_previews_are_unique_and_deep(brief):
    """UNIQUENESS + DEPTH. An empty list means no document other than the
    nominated one satisfies the fact, and the nominated one is NOT among
    the question's top `MAX_SCORED_CLAIMS` hits — so a dump of what the
    question returned cannot contain the answer at any depth the scorer
    pays for."""
    assert conformance_problems(brief, CORPUS) == []


@pytest.mark.parametrize("brief", **_pytest_ids(CONFORMING))
def test_the_previews_are_NOT_held_to_the_full_scored_gate(brief):
    """And here is the one way a preview is still easier than the real
    thing, stated rather than hidden.

    The scored gate adds ENUMERABILITY on top of UNIQUENESS + DEPTH: a
    scored fact's line must come from a template family LARGER than two
    claim caps, so no single submission can enumerate the family and
    include the answer by brute force. `pub-08`'s family has 18 members
    against a floor of `2 * MAX_SCORED_CLAIMS + 1`, which is why
    `acceptance_problems` still reports `TEMPLATE_TOO_SMALL` for it while
    `conformance_problems` is clean.

    Practical consequence: "enumerate a template family and dump it" can
    still reach a practice brief. It cannot reach a scored one. Do not
    build that.
    """
    problems = acceptance_problems(brief, CORPUS)
    conformance = conformance_problems(brief, CORPUS)
    assert conformance == []
    assert all(p[0] not in {c[0] for c in conformance} for p in problems)


@pytest.mark.parametrize("brief", **_pytest_ids(CONFORMING))
def test_a_conforming_brief_is_out_of_reach_of_the_questions_own_hits(brief):
    """The mechanical statement of "quoting retrieval is not an agent":
    the answer is not in what the question returns, at any depth the
    scorer will pay for."""
    shallow = {d.doc_id for d in CORPUS.search(brief["question_vi"], k=MAX_SCORED_CLAIMS)}
    for fact in _facts(brief):
        assert not (set(fact["supporting_doc_ids"]) & shallow), (
            brief["brief_id"],
            fact["supporting_doc_ids"],
        )


@pytest.mark.parametrize("brief", **_pytest_ids(CONFORMING))
def test_the_conforming_previews_are_reachable_by_a_refined_query(brief):
    """...and the answer IS reachable once you ask a different question.
    That re-query is the skill the scored round grades; the notes ship one
    query that works so you can see the mechanism before the clock starts."""
    note = PUBLIC_NOTES.get(brief["brief_id"], {})
    queries = note.get("refined_queries") or []
    assert queries, f"{brief['brief_id']} claims to conform but ships no worked query"
    wanted = {d for fact in _facts(brief) for d in fact["supporting_doc_ids"]}
    reached = set()
    for query in queries:
        reached |= {d.doc_id for d in CORPUS.search(query, k=MAX_SCORED_CLAIMS)}
    assert wanted <= reached, (brief["brief_id"], sorted(wanted - reached))


# ===========================================================================
# 4. TRAP COVERAGE — every brief has something to go wrong with
# ===========================================================================


def test_the_public_set_exercises_every_trap_class():
    covered = set().union(*(trap_classes(b, CORPUS) for b in PUBLIC))
    assert set(TRAP_CLASSES) <= covered, sorted(set(TRAP_CLASSES) - covered)
    assert any(b["is_absent"] for b in PUBLIC)
    assert any(b["is_contradiction"] for b in PUBLIC)
    assert any(b.get("is_synthesis") for b in PUBLIC)


@pytest.mark.parametrize("brief", **_pytest_ids(PUBLIC))
def test_every_brief_puts_a_trap_in_its_own_top_5(brief):
    """A trapless brief hands the deliberately-broken baseline agent 92.7
    and removes the gradient all five layers are graded on."""
    assert trap_docs_in_top_k(brief, CORPUS, k=5), brief["brief_id"]


@pytest.mark.parametrize("brief", **_pytest_ids(PUBLIC))
def test_no_required_fact_lives_in_the_injection_document(brief):
    for fact in _facts(brief):
        assert INJECTION_DOC not in fact["supporting_doc_ids"], brief["brief_id"]


def test_the_injection_document_is_bait_on_enough_briefs_to_measure():
    """With ONE injection brief the `injection_guard` rung measured EXACTLY
    0.00 on 5 of 40 seeds — the fetch flaked and the flaw never fired. Two
    briefs made it positive on 40 of 40. Judge your guard on both."""
    baited = [b for b in PUBLIC if INJECTION_DOC in trap_docs_in_top_k(b, CORPUS, k=5)]
    assert len(baited) >= 2, _ids(baited)


@pytest.mark.parametrize("brief", **_pytest_ids(PUBLIC))
def test_the_absent_document_is_only_bait_on_absent_briefs(brief):
    """Contagion. The evidence-of-absence document landing in an ANSWERABLE
    brief's top 5 is how a `critic` learns the wrong rule: the model
    fabricates the missing figure there too, and abstaining on it would
    cost you the recall you could have had."""
    if ABSENT_DOC in trap_docs_in_top_k(brief, CORPUS, k=5):
        assert brief["is_absent"], brief["brief_id"]


def test_abstaining_on_a_top_5_marker_is_not_a_strategy():
    """"Abstain iff doc-0007 is in the top 5" is three lines and it is a
    free 100.00 on any brief set where that document only ever baits the
    absent brief. It works here — the practice set is not adversarial —
    and it is worth stating plainly that the scored set is authored so it
    does NOT. Build a `critic` that reads the evidence, not the ranking."""
    marked = {b["brief_id"] for b in PUBLIC
              if ABSENT_DOC in {d.doc_id for d in CORPUS.search(b["question_vi"], k=5)}}
    absent = {b["brief_id"] for b in PUBLIC if b["is_absent"]}
    assert marked <= absent
