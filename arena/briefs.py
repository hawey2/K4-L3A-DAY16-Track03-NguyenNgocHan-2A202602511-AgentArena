"""Research briefs — the questions the arena grades you on, and the
mechanical checks that decide whether a brief is fit to be graded on.

A brief is a small JSON object:

    {
      "brief_id": "prv-01-...",
      "question_vi": "...",                    # what the agent is asked
      "required_facts": [                      # >= 1, ALWAYS
        {"claim": "<a verbatim quotation from the supporting document>",
         "supporting_doc_ids": ["doc-0069"],
         "key_terms": ["optional exact phrases that MUST appear"]}
      ],
      "is_absent": false,                      # mutually exclusive with
                                               # is_contradiction AND with
                                               # is_synthesis
      "is_contradiction": false,               # combines legally with
                                               # is_synthesis
      "is_synthesis": false,                   # OPTIONAL; declares a verdict
      "verdict": {                             # REQUIRED iff is_synthesis
        "options": [
          {"id": "ban_moi_thay_the",   "phrases": ["quy định mới thay thế..."]},
          {"id": "ban_cu_con_hieu_luc","phrases": ["quy định cũ vẫn còn..."]},
          {"id": "chua_du_can_cu",     "phrases": ["chưa đủ căn cứ để..."]}
        ],
        "correct": "ban_moi_thay_the",
        "requires_facts": [0, 1]               # optional; default = all
      },
      "budget": {"max_tool_calls": 8, "max_tokens": 12000, "max_seconds": 60.0}
    }

Two sets ship, and they are NOT interchangeable:

* ``data/briefs_public.json`` — the PRACTICE set. Unscored, advisory, and
  it ships its own ``required_facts`` so you can score yourself offline
  while you build. Seven of its eight briefs are deliberately authored
  the easy way (see LIFT-SPAN AUTHORING below) because ``MockModel``'s
  only quotable surface is ``arena.model._lift_span``; a brief the mock
  cannot answer is useless for practice. **Hard-coding the practice
  answers is legal and worthless**: the scored round runs on briefs you
  have never seen, and the two sets share no brief id, no question and no
  required-fact claim (pinned by ``tests/test_briefs.py``).
* ``instructor/briefs_private.json`` — the SCORED set. Not in the student
  bundle; :func:`load_private_briefs` raising ``FileNotFoundError`` is the
  normal, expected state on a student machine.

WHY BRIEF AUTHORING IS A SECURITY CONTROL, NOT CONTENT
------------------------------------------------------
``Corpus.search`` and ``arena.model._lift_span`` are frozen, public and
fifteen lines long, and the corpus generates from a published seed. So
"retrieve the top k, quote each hit's best lines, cite each to itself" is
an offline script — and on a real endpoint it also has genuine PROVENANCE
(ask the model to repeat what you just retrieved, and every byte of the
reply really is the model's). No rule inside ``arena/scorer.py`` reaches
that attack. Measured, on one brief:

    ground truth authored from ``_lift_span`` ............ 100.00
    ground truth in a line ``_lift_span`` misses ......... 100.00
    ground truth satisfying UNIQUENESS + DEPTH ...........  40.15

The brief is the defence, and it is a sufficient one. The properties that
make it stand are mechanical, and :func:`conformance_problems` checks
them:

1. **UNIQUENESS** — no document OTHER than the nominated one satisfies
   ``scorer._covers`` for the fact. The corpus is template-generated, so
   near-duplicates abound.
2. **DEPTH** — the supporting document is NOT among the question's top
   ``MAX_SCORED_CLAIMS`` hits, so a blind dump cannot include it inside
   the claim cap. It must still be REACHABLE by a refined query — that
   re-query is the skill being graded.
3. **THE VERDICT IS NOT QUOTABLE** (``is_synthesis`` briefs only) — no
   declared verdict phrase may appear in any line of any document. Both
   the supporting facts and UNIQUENESS/DEPTH still apply to them; what is
   new is the CONCLUSION those facts entail, which no document states.
   That is the one property retrieval depth cannot supply: an adaptive
   model-guided dumper reproduces any retrieval and then dumps, and it
   measured **97.28 mean on the private set** doing so. It cannot dump a
   line that does not exist, and reciting both candidates at once is
   priced at zero. :func:`verdict_problems` checks it.

UNIQUENESS and DEPTH are both defined against the brief's OWN question,
so neither sees an attacker that never asks one. Fix round 1 measured
what that costs: with all six answerable facts drawn from the SAME
18-document template family, a dump of ten lines picked by a fixed
corpus-only ORDERING scored 100.00 on four of the six private briefs and
65.58 blind across the set (rung 1 = 59.53) with every question-driven
check green. Two more mechanical properties close it:

4. **ENUMERABILITY** — the fact's line must come from a template family
   LARGER than two claim caps, and never from the hand-authored
   documents. :data:`MIN_TEMPLATE_FAMILY` carries the proof that a
   smaller family has no hiding place in it at all.
   :func:`enumerability_problems` checks it.
5. **DISPERSION** — no single question-independent enumeration may reach
   the facts of two different briefs. A recipe that guesses one brief's
   document right wins that brief — no authoring prevents that — but it
   must not win the round. :func:`dispersion_problems` checks it, over
   the whole set at once.

:func:`acceptance_problems` runs schema + UNIQUENESS/DEPTH + ENUMERABILITY
in one call, so an instructor can gate a brief without pytest.

``is_absent`` briefs are exempt from UNIQUENESS and DEPTH: the evidence
of absence is meant to be findable, and abstaining while citing it is the
correct answer rather than a cheat. The SHIPPED absent brief does not
take the DEPTH exemption: while its evidence sat in its own question's
top 5 and in no other brief's, "abstain iff that document is in the top
5" was a three-line free 100.00 on a sixth of the scored set.

LIFT-SPAN AUTHORING
-------------------
Authoring a fact from ``_lift_span(doc.body)`` of a shallow hit hands out
the answer key. It is allowed for the practice set ONLY, and every such
brief is labelled: ``conformance_problems`` returns a non-empty list for
it, on purpose, and ``tests/test_briefs.py`` asserts that it does.
"""

from __future__ import annotations

import json
import re
import weakref
from pathlib import Path

from arena.corpus import TRAP_CLASSES, Corpus
from arena.scorer import (
    MAX_SCORED_CLAIMS,
    _covers,
    _fact_terms,
    _norm,
    _norm_lines,
    _supports,
    _synthesis_spec,
    _WORD_RE,
)

__all__ = [
    "PUBLIC_BRIEFS_PATH",
    "PRIVATE_BRIEFS_PATH",
    "BRIEF_KEYS",
    "OPTIONAL_BRIEF_KEYS",
    "BUDGET_KEYS",
    "BriefSetError",
    "load_brief_file",
    "load_briefs",
    "load_notes",
    "load_public_briefs",
    "load_private_briefs",
    "brief_by_id",
    "schema_problems",
    "verdict_problems",
    "conformance_problems",
    "enumerability_problems",
    "dispersion_problems",
    "max_enumeration_reach",
    "DUMP_MISS_FLOOR",
    "RUNG_1_TOTAL",
    "acceptance_problems",
    "salient_enumerations",
    "line_template_family",
    "fact_line",
    "hand_authored_documents",
    "TEMPLATE_RUN_TOKENS",
    "MIN_TEMPLATE_FAMILY",
    "covering_documents",
    "key_term_documents",
    "fact_is_verbatim",
    "trap_docs_in_top_k",
    "trap_classes",
]

_LAB_ROOT = Path(__file__).resolve().parent.parent

#: The practice set. Ships to students.
PUBLIC_BRIEFS_PATH = _LAB_ROOT / "data" / "briefs_public.json"

#: The scored set. Instructor-only — absent from the student bundle.
PRIVATE_BRIEFS_PATH = _LAB_ROOT / "instructor" / "briefs_private.json"

#: Exactly the keys a brief carries. Extra keys are a schema error: the
#: scorer ignores what it does not know, so a typo'd flag would silently
#: grade the wrong thing.
BRIEF_KEYS = (
    "brief_id",
    "question_vi",
    "required_facts",
    "is_absent",
    "is_contradiction",
    "budget",
)

#: Keys a brief MAY carry. `is_synthesis` is opt-in — every brief authored
#: before the class existed is still well formed without it — and
#: `verdict` is required exactly when `is_synthesis` is true. Kept
#: separate from `BRIEF_KEYS` on purpose: that tuple is also the
#: REQUIRED-key list, and moving these into it would make every existing
#: brief report a missing key.
OPTIONAL_BRIEF_KEYS = ("is_synthesis", "verdict")

BUDGET_KEYS = ("max_tool_calls", "max_tokens", "max_seconds")


class BriefSetError(ValueError):
    """A brief file is missing, malformed, or not the set it claims to be."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_brief_file(path) -> dict:
    """Read one brief file whole: ``{set, corpus_seed, briefs, instructor_notes}``.

    Raises :class:`BriefSetError` rather than returning something empty —
    a scored round that silently grades zero briefs is worse than one
    that refuses to start.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:  # pragma: no cover - unreadable file
        raise BriefSetError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BriefSetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("briefs"), list):
        raise BriefSetError(f"{path} does not carry a 'briefs' list")
    if not payload["briefs"]:
        raise BriefSetError(f"{path} carries no briefs")
    return payload


def load_briefs(path) -> list[dict]:
    """The briefs in `path`, in file order (which is the run order)."""
    return list(load_brief_file(path)["briefs"])


def load_notes(path) -> dict:
    """Instructor notes keyed by ``brief_id`` (refined queries, trap docs).

    Notes are documentation and test fixtures. Nothing in the scoring
    path reads them.
    """
    notes = load_brief_file(path).get("instructor_notes")
    return dict(notes) if isinstance(notes, dict) else {}


def load_public_briefs() -> list[dict]:
    return load_briefs(PUBLIC_BRIEFS_PATH)


def load_private_briefs(path=None) -> list[dict]:
    """The scored set. ``FileNotFoundError`` on a student machine is normal."""
    return load_briefs(PRIVATE_BRIEFS_PATH if path is None else path)


def brief_by_id(briefs, brief_id: str) -> dict:
    for brief in briefs:
        if brief.get("brief_id") == brief_id:
            return brief
    raise KeyError(brief_id)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def schema_problems(brief) -> list[str]:
    """Everything structurally wrong with `brief`. Empty list == well formed."""
    problems: list[str] = []
    if not isinstance(brief, dict):
        return ["brief is not an object"]

    missing = [k for k in BRIEF_KEYS if k not in brief]
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
    extra = [k for k in brief if k not in BRIEF_KEYS + OPTIONAL_BRIEF_KEYS]
    if extra:
        problems.append(f"unknown keys: {sorted(extra)}")

    if not isinstance(brief.get("brief_id"), str) or not brief.get("brief_id"):
        problems.append("brief_id must be a non-empty string")
    if not isinstance(brief.get("question_vi"), str) or not brief.get("question_vi"):
        problems.append("question_vi must be a non-empty string")

    for flag in ("is_absent", "is_contradiction"):
        if not isinstance(brief.get(flag), bool):
            problems.append(f"{flag} must be a bool")
    if brief.get("is_absent") is True and brief.get("is_contradiction") is True:
        # Measured: setting both let the contradiction branch win the
        # honesty test, so a CONFIDENT answer on an absent brief scored
        # 100.00 instead of 83.68. The scorer flags it; the brief must
        # never rely on that.
        problems.append("is_absent and is_contradiction are mutually exclusive")

    # --- SYNTHESIS. Opt-in, and the declaration is parsed by the SCORER's
    # own parser so this check and the grade can never disagree about what
    # a well-formed verdict is.
    synthesis = brief.get("is_synthesis")
    if synthesis is not None and not isinstance(synthesis, bool):
        problems.append("is_synthesis must be a bool")
    if "verdict" in brief and synthesis is not True:
        problems.append("verdict is declared but is_synthesis is not true")
    if synthesis is True:
        if brief.get("is_absent") is True:
            # "there is no answer" and "here is the answer you must derive"
            # cannot both hold.
            problems.append("is_absent and is_synthesis are mutually exclusive")
        if "verdict" not in brief:
            problems.append("is_synthesis briefs must declare a verdict")
        problems.extend(_synthesis_spec(brief)[1])

    facts = brief.get("required_facts")
    if not isinstance(facts, list) or not facts:
        problems.append("required_facts must be a non-empty list")
    else:
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                problems.append(f"required_facts[{index}] is not an object")
                continue
            claim = fact.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                problems.append(f"required_facts[{index}].claim must be a non-empty string")
            docs = fact.get("supporting_doc_ids")
            if not isinstance(docs, list) or not docs or not all(
                isinstance(d, str) and d for d in docs
            ):
                problems.append(
                    f"required_facts[{index}].supporting_doc_ids must be a non-empty list of ids"
                )
            terms = fact.get("key_terms")
            if terms is not None and (
                not isinstance(terms, list)
                or not terms
                or not all(isinstance(t, str) and t.strip() for t in terms)
            ):
                problems.append(f"required_facts[{index}].key_terms must be a non-empty list")
            unknown = [k for k in fact if k not in ("claim", "supporting_doc_ids", "key_terms")]
            if unknown:
                problems.append(f"required_facts[{index}] has unknown keys: {sorted(unknown)}")
        if brief.get("is_contradiction") is True and len(facts) < 2:
            # One fact per conflicting document is what lets the scorer
            # tell "surfaced the conflict" from "picked a side".
            problems.append("is_contradiction briefs must declare both sides as facts")

    budget = brief.get("budget")
    if not isinstance(budget, dict):
        problems.append("budget must be an object")
    else:
        for key in BUDGET_KEYS:
            value = budget.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                problems.append(f"budget.{key} must be a positive number")
        unknown = [k for k in budget if k not in BUDGET_KEYS]
        if unknown:
            problems.append(f"budget has unknown keys: {sorted(unknown)}")
    return problems


# ---------------------------------------------------------------------------
# The two mechanical properties
# ---------------------------------------------------------------------------


def covering_documents(claim: str, question: str, corpus: Corpus) -> list[str]:
    """Every document that satisfies ``scorer._covers`` for `claim`.

    Uses the CLAIM, never ``key_terms`` — deliberately, and identically to
    the check shipped in ``tests/test_scorer.py``: uniqueness is a property
    of the authored sentence, and authoring a loose ``key_terms`` must not
    be able to buy a brief past this test.
    """
    terms = _fact_terms({"claim": claim}, question)
    hits = []
    for doc in corpus.docs:
        normalised = _norm(doc.body)
        if _covers(set(_WORD_RE.findall(normalised)), normalised, terms):
            hits.append(doc.doc_id)
    return hits


def key_term_documents(fact, corpus: Corpus) -> list[str]:
    """Documents whose body contains EVERY ``key_terms`` phrase.

    ``key_terms`` overrides the claim in the scorer's recall test, so a
    fact that pins them opens a second surface that
    :func:`conformance_problems` does not see. This is the check for that
    surface; ``tests/test_briefs.py`` asserts it is as tight as the
    claim-derived one. Returns ``[]`` for a fact with no ``key_terms``.
    """
    terms = fact.get("key_terms")
    if not isinstance(terms, list) or not terms:
        return []
    wanted = [_norm(t) for t in terms if isinstance(t, str) and t.strip()]
    if not wanted:
        return []
    return [
        doc.doc_id
        for doc in corpus.docs
        if all(phrase in _norm(doc.body) for phrase in wanted)
    ]


def fact_is_verbatim(fact, corpus: Corpus) -> bool:
    """True if the claim is a quotation of ONE LINE of a supporting document.

    The same test the scorer applies (``_supports``): support is scoped to
    a line, so a splice across a title, a blank line and half a paragraph
    is not a quotation and a brief must never demand one.
    """
    claim = fact.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        return False
    normalised = _norm(claim)
    for doc_id in fact.get("supporting_doc_ids") or []:
        doc = corpus.get(doc_id)
        if doc is not None and _supports(_norm_lines(doc.body), normalised):
            return True
    return False


def verdict_problems(brief, corpus: Corpus) -> list[tuple]:
    """The synthesis class's own mechanical check. Empty list == fit.

    A synthesis brief's VERDICT must exist verbatim in NO document line.
    That is not a stylistic preference, it is the entire class: a verdict
    a document says is a required fact wearing a hat, and once it is
    quotable the adaptive dumper that measured **97.28** states it for
    free. The rule bites in the other direction too, and that half is
    what would actually reach a student — if a verdict phrase sat in a
    document line, an honest agent quoting its own evidence would assert
    a candidate it never chose, hedge, and lose the slot it earned.

    Structural validity comes from the SCORER's own parser, so this check
    and the grade can never disagree about what a well-formed verdict is.

    NOT rejected, but worth saying: a TWO-option verdict is legal and a
    coin flip is worth 1/2 of one slot. Prefer three or more, and ship
    several synthesis briefs, so guessing does not survive aggregation.
    """
    problems: list[tuple] = []
    if brief.get("is_synthesis") is not True:
        return problems

    spec, malformed = _synthesis_spec(brief)
    for problem in malformed:
        problems.append(("VERDICT_MALFORMED", problem, ()))
    if spec is None:
        return problems

    for option in spec["options"]:
        for phrase in option["phrases"]:
            for doc_id in corpus.doc_ids():
                doc = corpus.get(doc_id)
                if doc is None:
                    continue
                if any(phrase in line for line in _norm_lines(doc.body)):
                    problems.append(
                        ("VERDICT_IS_A_DOCUMENT_LINE", option["id"], (phrase, doc_id))
                    )
                    break
    return problems


def conformance_problems(brief, corpus: Corpus) -> list[tuple]:
    """UNIQUENESS + DEPTH, plus the synthesis class's verdict check.
    Empty list == fit for the SCORED set.

    Mirrors ``tests/test_scorer.brief_conformance_problems`` exactly —
    ``tests/test_briefs.py`` pins the two to the same verdict on every
    shipped brief, so this copy cannot drift away from the contract it
    implements. Answerable briefs only: an ``is_absent`` brief's fact is
    the evidence of absence and is meant to be findable.

    The verdict check runs BEFORE the ``is_absent`` exemption, because a
    brief that sets both flags is malformed in a way the exemption would
    otherwise hide.
    """
    problems: list[tuple] = list(verdict_problems(brief, corpus))
    if brief.get("is_absent"):
        return problems
    question = brief.get("question_vi") or ""
    shallow = {d.doc_id for d in corpus.search(question, k=MAX_SCORED_CLAIMS)}
    for fact in brief.get("required_facts", []):
        claim = fact.get("claim", "")
        nominated = list(fact.get("supporting_doc_ids", []))
        covering = covering_documents(claim, question, corpus)
        if covering != nominated:
            problems.append(("NOT_UNIQUE", claim[:48], covering))
        if any(d in shallow for d in nominated):
            problems.append(("TOO_SHALLOW", claim[:48], nominated))
    return problems


# ---------------------------------------------------------------------------
# ENUMERABILITY — the surface UNIQUENESS + DEPTH cannot see
# ---------------------------------------------------------------------------
#
# DEPTH is defined against the brief's OWN question, so it says nothing at
# all about an attacker who never reads the question. Measured on the first
# draft of the private set, whose six answerable facts all came from the
# SAME 18-document template family: a dump of ten policy lines picked by a
# fixed corpus-only ordering scored 100.00 on five of six briefs and 65.58
# blind across the set, against rung 1 at 59.53. Nothing in
# `conformance_problems` sees that, because the attack never searches the
# question.
#
# Two properties close it, and both are mechanical:
#
# 1. **ENUMERABILITY** (per fact). The fact's line must come from a
#    template family LARGER than two claim caps, and never from the
#    hand-authored documents. See `MIN_TEMPLATE_FAMILY` for the proof that
#    a smaller family has no hiding place in it at all.
# 2. **DISPERSION** (per set). No single question-independent enumeration
#    may reach the facts of two different briefs. A dump that guesses one
#    brief's document right wins THAT brief; it must not win the round.
#
# What these do NOT claim, because it is measurably false: that no cheap
# enumeration reaches a fact. Twenty orderings of ten exhaust a
# thirty-five document family, so for ANY authoring choice some ordering
# contains the answer. The residual is priced in `tests/test_briefs.py`
# (per-shape set means, and a distribution over random dumps) rather than
# asserted away.

#: Tokens of contiguous overlap that make two lines instances of one
#: TEMPLATE. Every filler line in this corpus is one fixed sentence with a
#: few slots, so an eight-token shared run is a template match, not a
#: coincidence: it groups the shipped corpus into the policy line's 18
#: documents and the report line's 35, and leaves the eight hand-authored
#: documents in families of one.
TEMPLATE_RUN_TOKENS = 8

#: The smallest template family a SCORED fact may be drawn from.
#: DERIVED, not chosen: one submission may quote `MAX_SCORED_CLAIMS`
#: documents. In a family of size F, a document at rank r of some ordering
#: sits at rank F+1-r in the reverse of that ordering, so if
#: F <= 2*MAX_SCORED_CLAIMS then min(r, F+1-r) <= MAX_SCORED_CLAIMS for
#: EVERY member: whichever document the fact is drawn from, the first ten
#: of an ordering or of its reverse already contains it. The corpus's 18
#: policy documents are such a family.
MIN_TEMPLATE_FAMILY = 2 * MAX_SCORED_CLAIMS + 1

#: Shortest line worth treating as quotable evidence (the scorer will not
#: credit a claim below `MIN_SUPPORT_CHARS` either).
MIN_LINE_CHARS = 12

_DIGIT_RUN_RE = re.compile(r"\d+")

_INDEX_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _index(corpus: Corpus) -> dict:
    """Per-corpus derived data (lines, normalised bodies). A pure cache."""
    try:
        cached = _INDEX_CACHE.get(corpus)
    except TypeError:  # pragma: no cover - a corpus that cannot be weak-ref'd
        cached = None
    if cached is not None:
        return cached
    lines = {}
    for doc in corpus.docs:
        lines[doc.doc_id] = tuple(
            line
            for line in (raw.strip() for raw in doc.body.splitlines())
            if len(line) >= MIN_LINE_CHARS
        )
    index = {
        "lines": lines,
        "bodies": {doc.doc_id: _norm(doc.body) for doc in corpus.docs},
        "order": [doc.doc_id for doc in corpus.docs],
    }
    try:
        _INDEX_CACHE[corpus] = index
    except TypeError:  # pragma: no cover
        pass
    return index


def _numbers(text: str) -> list[int]:
    return [int(match) for match in _DIGIT_RUN_RE.findall(text or "")]


def fact_line(fact, corpus: Corpus) -> str:
    """The DOCUMENT LINE the fact's claim was lifted from.

    Templates are a property of the line, not of the sentence an author
    chose to quote out of it: a claim that stops before the template's
    invariant tail would otherwise look like a family of three.
    """
    claim = _norm(fact.get("claim") or "")
    if not claim:
        return ""
    for doc_id in fact.get("supporting_doc_ids") or []:
        for line in _index(corpus)["lines"].get(doc_id, ()):
            if claim in _norm(line):
                return line
    return fact.get("claim") or ""


def line_template_family(line: str, corpus: Corpus) -> list[str]:
    """Documents carrying a line built from the same TEMPLATE as `line`.

    Membership is `TEMPLATE_RUN_TOKENS` tokens of contiguous overlap, which
    is what a slot-substituted sentence shares with its siblings. Sorted,
    so the result is identical across processes.
    """
    tokens = _WORD_RE.findall(_norm(line or ""))
    if len(tokens) < TEMPLATE_RUN_TOKENS:
        return []
    runs = [
        " ".join(tokens[i:i + TEMPLATE_RUN_TOKENS])
        for i in range(len(tokens) - TEMPLATE_RUN_TOKENS + 1)
    ]
    bodies = _index(corpus)["bodies"]
    return sorted(
        doc_id for doc_id, body in bodies.items() if any(run in body for run in runs)
    )


def hand_authored_documents(corpus: Corpus) -> list[str]:
    """Documents whose prose belongs to no template family — the eight
    trap documents of the shipped corpus, derived rather than hard-coded.

    They are the WORST place for a required fact: a single submission can
    quote every one of them, so a fact hosted there is free to a dump that
    reads nothing.
    """
    lines = _index(corpus)["lines"]
    found = []
    for doc_id in _index(corpus)["order"]:
        if all(
            len(line_template_family(line, corpus)) <= 1
            for line in lines.get(doc_id, ())
        ):
            found.append(doc_id)
    return found


# --- the line an attacker would quote out of each document ----------------


def _template_line(doc_id: str, reference: str, corpus: Corpus):
    """The `reference` line's sibling inside `doc_id`, if it has one."""
    tokens = _WORD_RE.findall(_norm(reference or ""))
    if len(tokens) < TEMPLATE_RUN_TOKENS:
        return None
    runs = [
        " ".join(tokens[i:i + TEMPLATE_RUN_TOKENS])
        for i in range(len(tokens) - TEMPLATE_RUN_TOKENS + 1)
    ]
    for line in _index(corpus)["lines"].get(doc_id, ()):
        normalised = _norm(line)
        if any(run in normalised for run in runs):
            return line
    return None


def _digit_line(doc_id: str, corpus: Corpus):
    """The longest digit-bearing, non-pipe line — what `_lift_span` picks."""
    lines = [
        line
        for line in _index(corpus)["lines"].get(doc_id, ())
        if any(ch.isdigit() for ch in line) and "|" not in line
    ]
    return max(lines, key=lambda line: (len(line), line)) if lines else None


def _longest_line(doc_id: str, corpus: Corpus):
    lines = _index(corpus)["lines"].get(doc_id, ())
    return max(lines, key=lambda line: (len(line), line)) if lines else None


def _selectors(reference: str, corpus: Corpus) -> dict:
    return {
        "template_line": lambda doc_id: _template_line(doc_id, reference, corpus),
        "digit_line": lambda doc_id: _digit_line(doc_id, corpus),
        "longest_line": lambda doc_id: _longest_line(doc_id, corpus),
    }


def _orderings(pool: list, line_of, corpus: Corpus) -> dict:
    """Question-independent orderings of `pool`, derived from the documents.

    The keys are the ones an attacker has a REASON to write without an
    answer key: the corpus's own order, and each numeric slot of the line
    it is about to quote (the largest/smallest figure in the family). Both
    directions, because reversing a sort is free.

    Orderings keyed on something arbitrary — a digest, an author name, a
    body length — are deliberately NOT here. They are not a strategy, they
    are a random subset of the family, and no authoring choice escapes all
    of them (measured: over a 35-document family, 20 such orderings of 10
    cover every member). `tests/test_briefs.py` prices that residual with
    a distribution over random dumps instead of pretending it is closed.
    """
    keyed = [(doc_id, line_of(doc_id)) for doc_id in pool]
    keyed = [(doc_id, line) for doc_id, line in keyed if line]
    if not keyed:
        return {}
    orders = {"doc_id": lambda item: item[0]}
    slots = min(len(_numbers(line)) for _, line in keyed)
    for slot in range(min(slots, 3)):
        orders[f"number{slot}"] = (
            lambda item, slot=slot: (_numbers(item[1])[slot], item[0])
        )
    built = {}
    for name, key in orders.items():
        ascending = sorted(keyed, key=key)
        built[f"{name}+"] = ascending
        built[f"{name}-"] = list(reversed(ascending))
    return built


def salient_enumerations(briefs, corpus: Corpus, prefix: int = MAX_SCORED_CLAIMS) -> dict:
    """``{name: ((claim, doc_id), ...)}`` — every question-independent dump
    an attacker has a reason to submit, capped at `prefix` claims.

    The pools are the template families the SET's own facts live in (an
    attacker who targets anything else cannot reach them by construction),
    plus the hand-authored documents, plus the corpus in its own order.
    """
    pools: dict = {}
    for brief in briefs:
        for fact in brief.get("required_facts", []):
            line = fact_line(fact, corpus)
            family = line_template_family(line, corpus)
            if len(family) > 1:
                pools[f"family[{len(family)}]"] = (family, line)
    pools["hand_authored"] = (hand_authored_documents(corpus), "")
    pools["corpus"] = (list(_index(corpus)["order"]), "")

    dumps: dict = {}
    for pool_name, (pool, reference) in sorted(pools.items()):
        for sel_name, select in _selectors(reference, corpus).items():
            for order_name, ordered in _orderings(pool, select, corpus).items():
                claims = [(line, doc_id) for doc_id, line in ordered[:prefix]]
                for size in range(1, len(claims) + 1):
                    dumps[f"{pool_name}/{sel_name}/{order_name}/n={size}"] = tuple(
                        claims[:size]
                    )
    return dumps


def enumerability_problems(brief, corpus: Corpus) -> list[tuple]:
    """ENUMERABILITY, per fact. Empty list == not cheaply enumerable.

    `is_absent` briefs are exempt for the same reason the contract exempts
    them from UNIQUENESS + DEPTH: the evidence of absence lives in the one
    hand-authored document that says the data is missing, and it has to be
    findable. Their exposure is priced explicitly in `tests/test_briefs.py`
    instead — see the `is_absent` section there.
    """
    problems: list[tuple] = []
    if brief.get("is_absent"):
        return problems
    hand_authored = set(hand_authored_documents(corpus))
    for fact in brief.get("required_facts", []):
        claim = fact.get("claim", "")
        family = line_template_family(fact_line(fact, corpus), corpus)
        nominated = list(fact.get("supporting_doc_ids", []))
        if any(doc_id in hand_authored for doc_id in nominated):
            problems.append(("HAND_AUTHORED", claim[:48], sorted(set(nominated) & hand_authored)))
        elif len(family) < MIN_TEMPLATE_FAMILY:
            problems.append(("TEMPLATE_TOO_SMALL", claim[:48], len(family)))
    return problems


#: What a dump that reaches NO required fact banks across the set —
#: three policy lines, cited correctly, quoted verbatim, answering
#: nothing. MEASURED on this contract by
#: ``tests/test_briefs.py::miss_floor`` (37.43 answering, 45.36
#: abstaining on everything). It is the honest comparator for "did the
#: dump's claims earn anything at all".
DUMP_MISS_FLOOR = 37.43

#: Rung 1 of the BINDING ladder — the trap-spanning six-brief set with
#: one layer (`citation_checker`) installed. An entrant that banks this
#: with a dump has won the round against a student who built a layer.
RUNG_1_TOTAL = 59.53


def max_enumeration_reach(n_answerable: int) -> int:
    """How many briefs ONE question-independent enumeration may reach.

    DERIVED, not chosen, and derived from the sentence
    :func:`dispersion_problems` has always been enforcing: *a recipe that
    guesses one brief's document right wins that brief — no authoring
    prevents that — but it must not win the ROUND.*

    A dump that wins `r` of `n` briefs and misses the rest banks about
    ``(r * 100 + (n - r) * DUMP_MISS_FLOOR) / n``. The bound is the
    largest `r` for which that stays at or under :data:`RUNG_1_TOTAL`.

    On a SIX-brief set this returns 1, which is exactly the constant this
    function replaces — the rule has not been loosened, it has been
    written down in the units it was always in. On an ELEVEN-brief set it
    returns 3, because a lucky recipe that wins three of eleven banks
    54.50 and a student with one layer banks 59.53. Reaching one brief in
    six is worth twice what reaching one in eleven is worth, and a
    constant cannot see that.

    Two things this is NOT:

    * It is not the binding gate. The binding gate is MEASURED —
      ``tests/test_briefs.py::test_no_salient_dump_beats_rung_1_across_the_set``
      scores every recipe against the real scorer, where a multi-fact
      brief pays a dump partial recall and a synthesis brief pays it no
      verdict at all. This function models each reached brief as a full
      100, which is the worst case and strictly pessimistic.
    * It is not a licence to pack briefs together. `r` is a CEILING on an
      accident, not a budget to spend: every pair that shares a recipe is
      still correlation, and `tests/test_briefs.py` reports the reach
      distribution rather than only its maximum.
    """
    if n_answerable <= 0:
        return 1
    reach = 0
    while reach < n_answerable:
        banked = (reach + 1) * 100.0 + (n_answerable - reach - 1) * DUMP_MISS_FLOOR
        if banked / n_answerable > RUNG_1_TOTAL:
            break
        reach += 1
    return max(1, reach)


def dispersion_problems(briefs, corpus: Corpus) -> list[tuple]:
    """DISPERSION, per SET. Empty list == no single dump wins the round.

    A question-independent dump that happens to contain one brief's
    supporting document scores full marks on that brief — nothing in the
    scorer can tell it from an answer. What must not happen is one dump
    reaching ENOUGH of the set to bank a passing total, because that is
    what turns a lottery ticket into a leaderboard position: the first
    draft of this set had every fact in one 18-document family, where a
    single five-claim dump reached four of six briefs.

    "Enough" is :func:`max_enumeration_reach` — 1 on a six-brief set (the
    constant this replaced), 3 on an eleven-brief one. See that function
    for why a constant is the wrong unit and why this is a necessary
    condition rather than the binding one.
    """
    problems: list[tuple] = []
    covers: dict = {}
    for brief in briefs:
        if brief.get("is_absent"):
            continue
        question = brief.get("question_vi") or ""
        wanted = []
        for fact in brief.get("required_facts", []):
            terms = _fact_terms(fact, question)
            wanted.append(terms)
        covers[brief.get("brief_id")] = wanted
    cap = max_enumeration_reach(len(covers))
    for name, claims in salient_enumerations(briefs, corpus).items():
        reached = []
        for brief_id, facts in covers.items():
            texts = [_norm(text) for text, _ in claims]
            if any(
                _covers(set(_WORD_RE.findall(text)), text, terms)
                for terms in facts
                for text in texts
            ):
                reached.append(brief_id)
        if len(reached) > cap:
            problems.append(("SHARED_ENUMERATION", name, sorted(reached)))
    return problems


def acceptance_problems(brief, corpus: Corpus) -> list[tuple]:
    """Everything that makes `brief` unfit for the SCORED set, in one call.

    schema + UNIQUENESS/DEPTH (the frozen contract) + ENUMERABILITY (this
    module). Set-level DISPERSION is `dispersion_problems`, which needs all
    the briefs at once.
    """
    return (
        [("SCHEMA", problem, ()) for problem in schema_problems(brief)]
        + list(conformance_problems(brief, corpus))
        + list(enumerability_problems(brief, corpus))
    )


# ---------------------------------------------------------------------------
# Traps
# ---------------------------------------------------------------------------


def trap_docs_in_top_k(brief, corpus: Corpus, k: int = 5) -> dict:
    """``{doc_id: (trap tags,)}`` for the trap documents the question's own
    top-`k` returns.

    A brief with NO trap in its top-`k` is a brief the deliberately-broken
    baseline agent answers correctly (measured 92.7), which removes the
    whole gradient the five layers are graded on.
    """
    question = brief.get("question_vi") or ""
    hits = corpus.search(question, k=k)
    found = {}
    for doc in hits:
        traps = tuple(t for t in doc.tags if t in TRAP_CLASSES)
        if traps:
            found[doc.doc_id] = traps
    return found


def trap_classes(brief, corpus: Corpus, k: int = 5) -> set:
    """Trap classes reachable from this brief — by retrieval, and by kind.

    ``is_absent`` / ``is_contradiction`` are counted as their classes even
    when the retrieval bait sits elsewhere, because the brief exercises
    the layer either way.
    """
    classes = {t for traps in trap_docs_in_top_k(brief, corpus, k).values() for t in traps}
    if brief.get("is_absent"):
        classes.add("absent")
    if brief.get("is_contradiction"):
        classes.add("contradiction")
    return classes
