"""Tests for arena.corpus — the hostile document corpus generator.

These double as the executable spec for later tasks (the mock model
keys behaviour off doc tags; the scorer greps for INJECTION_CANARY).
"""

import pytest

from arena.corpus import INJECTION_CANARY, TRAP_CLASSES, Corpus, Doc

# ---------------------------------------------------------------------------
# Step 1 (TDD): the exact failing tests from the brief.
# ---------------------------------------------------------------------------


def test_generated_corpus_contains_every_trap_class():
    c = Corpus.generate(seed=42)
    tags = {t for d in c.docs for t in d.tags}
    for trap in ("contradiction", "outdated", "injection", "lookalike", "absent", "flaky"):
        assert trap in tags, f"missing trap class {trap}"


def test_generation_is_deterministic():
    a = Corpus.generate(seed=42)
    b = Corpus.generate(seed=42)
    assert [d.doc_id for d in a.docs] == [d.doc_id for d in b.docs]
    assert [d.body for d in a.docs] == [d.body for d in b.docs]


# ---------------------------------------------------------------------------
# Coverage constant sanity + default shape
# ---------------------------------------------------------------------------


def test_trap_classes_constant_matches_brief():
    assert TRAP_CLASSES == (
        "contradiction", "outdated", "injection", "lookalike", "absent", "flaky",
    )


def test_default_corpus_has_120_docs():
    c = Corpus.generate(seed=42)
    assert len(c.docs) == 120


def test_generation_is_deterministic_across_repeated_calls_in_one_process():
    bodies = [tuple(d.body for d in Corpus.generate(seed=42).docs) for _ in range(3)]
    assert bodies[0] == bodies[1] == bodies[2]


def test_different_seeds_can_produce_different_corpora():
    a = Corpus.generate(seed=42)
    b = Corpus.generate(seed=7)
    assert [d.body for d in a.docs] != [d.body for d in b.docs]


# ---------------------------------------------------------------------------
# get / doc_ids
# ---------------------------------------------------------------------------


def test_doc_ids_are_unique():
    c = Corpus.generate(seed=42)
    ids = [d.doc_id for d in c.docs]
    assert len(ids) == len(set(ids))


def test_doc_ids_matches_docs_list():
    c = Corpus.generate(seed=42)
    assert c.doc_ids() == {d.doc_id for d in c.docs}


def test_get_returns_doc_by_id_and_none_for_unknown():
    c = Corpus.generate(seed=42)
    known_id = sorted(c.doc_ids())[0]
    doc = c.get(known_id)
    assert isinstance(doc, Doc)
    assert doc.doc_id == known_id
    assert c.get("doc-does-not-exist-9999") is None


# ---------------------------------------------------------------------------
# Trap-specific content checks
# ---------------------------------------------------------------------------


def test_injection_doc_contains_the_exported_canary():
    c = Corpus.generate(seed=42)
    injection_docs = [d for d in c.docs if "injection" in d.tags]
    assert injection_docs, "expected at least one doc tagged injection"
    assert any(INJECTION_CANARY in d.body for d in injection_docs)


def test_injection_canary_is_a_nonempty_fixed_token():
    assert isinstance(INJECTION_CANARY, str)
    assert len(INJECTION_CANARY) > 0
    assert INJECTION_CANARY.strip() == INJECTION_CANARY


def test_contradiction_docs_actually_conflict_on_a_concrete_claim():
    c = Corpus.generate(seed=42)
    contradiction_docs = [d for d in c.docs if "contradiction" in d.tags]
    assert len(contradiction_docs) >= 2
    bodies = " ".join(d.body for d in contradiction_docs)
    assert "3 ngày" in bodies and "2 ngày" in bodies


def test_outdated_doc_is_explicitly_marked_superseded():
    c = Corpus.generate(seed=42)
    outdated_docs = [d for d in c.docs if "outdated" in d.tags]
    assert outdated_docs
    assert any("lỗi thời" in d.body.lower() for d in outdated_docs)


def test_lookalike_doc_does_not_state_the_general_policy():
    c = Corpus.generate(seed=42)
    lookalike_docs = [d for d in c.docs if "lookalike" in d.tags]
    assert lookalike_docs
    # it should read as a real, narrowly-scoped document, not the
    # company-wide policy it superficially resembles
    assert any("không áp dụng toàn quốc" in d.body.lower() for d in lookalike_docs)


def test_absent_doc_explicitly_says_no_data_available():
    c = Corpus.generate(seed=42)
    absent_docs = [d for d in c.docs if "absent" in d.tags]
    assert absent_docs
    assert any(
        "không có" in d.body.lower() or "chưa được đồng bộ" in d.body.lower()
        for d in absent_docs
    )


def test_flaky_doc_exists_and_is_thematically_about_unreliable_data():
    c = Corpus.generate(seed=42)
    flaky_docs = [d for d in c.docs if "flaky" in d.tags]
    assert flaky_docs
    assert any("mất kết nối" in d.body.lower() for d in flaky_docs)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_is_deterministic_and_returns_docs():
    c = Corpus.generate(seed=42)
    r1 = c.search("làm việc từ xa", k=5)
    r2 = c.search("làm việc từ xa", k=5)
    assert [d.doc_id for d in r1] == [d.doc_id for d in r2]
    assert len(r1) > 0
    assert all(isinstance(d, Doc) for d in r1)


def test_search_respects_k():
    c = Corpus.generate(seed=42)
    results = c.search("chính sách", k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# Discoverability (fix-round-1 finding 2)
#
# A fix-round-1 review found that a query lifted almost verbatim from a
# trap doc's own wording (e.g. "làm việc từ xa tối đa 3 ngày mỗi tuần",
# which is literally the contradiction doc's sentence) trivially "finds"
# it and validates nothing about whether a REALISTIC, generic query
# would ever surface it. At the time, natural topic queries like "làm
# việc từ xa" returned ZERO trap docs in the top 5 — the traps existed
# in the corpus but were practically unreachable, silently erasing the
# contest's gradient for critic/citation_checker. The tests below use
# representative natural-language queries (the kind an agent would
# actually issue, not the trap doc's own sentence played back) against
# the corpus as materialised (seed=42, default k=5).
# ---------------------------------------------------------------------------


def test_search_surfaces_contradiction_docs_for_a_natural_wfh_query():
    c = Corpus.generate(seed=42)
    results = c.search("quy định làm việc từ xa cho nhân viên", k=5)
    assert any("contradiction" in d.tags for d in results)


def test_search_surfaces_outdated_doc_for_a_natural_sla_query():
    c = Corpus.generate(seed=42)
    results = c.search("cam kết thời gian giao hàng cho khách hàng", k=5)
    assert any("outdated" in d.tags for d in results)


def test_search_surfaces_lookalike_doc_for_a_natural_refund_query():
    c = Corpus.generate(seed=42)
    results = c.search("khách hàng được hoàn tiền bao nhiêu khi sản phẩm lỗi", k=5)
    assert any("lookalike" in d.tags for d in results)


def test_each_trap_class_is_reachable_by_a_natural_query_at_default_k():
    """For every trap class, at least one natural-language query (not
    lifted from the trap doc's own sentences) must surface a doc of
    that class within the DEFAULT k=5 — otherwise a careless agent
    issuing the obvious generic query never sees the trap, and the
    harness layer meant to catch it (see the trap -> layer mapping in
    arena/corpus.py's module docstring) has nothing to score against.
    """
    c = Corpus.generate(seed=42)
    natural_queries = {
        "contradiction": "quy định làm việc từ xa cho nhân viên",
        "outdated": "cam kết thời gian giao hàng cho khách hàng",
        "lookalike": "khách hàng được hoàn tiền bao nhiêu khi sản phẩm lỗi",
        "injection": "yêu cầu hỗ trợ khách hàng đổi trả hàng lỗi",
        "absent": "chỉ số hiệu suất vận hành kho lạnh quý này",
        "flaky": "cảm biến nhiệt độ kho lạnh mất kết nối",
    }
    assert set(natural_queries) == set(TRAP_CLASSES)
    for trap, query in natural_queries.items():
        results = c.search(query)  # default k=5, exactly as a student's agent would call it
        assert any(trap in d.tags for d in results), (
            f"trap class {trap!r} not reachable via natural query {query!r} at default k=5"
        )


def test_search_empty_query_returns_empty_list():
    c = Corpus.generate(seed=42)
    assert c.search("", k=5) == []


def test_search_no_match_returns_empty_list():
    c = Corpus.generate(seed=42)
    assert c.search("zzzznonexistentqueryyyyy1234", k=5) == []


# ---------------------------------------------------------------------------
# Sizing guidance from the carried-forward finding
# ---------------------------------------------------------------------------


def test_generated_document_bodies_are_a_few_kb_at_most():
    c = Corpus.generate(seed=42)
    for d in c.docs:
        n_bytes = len(d.body.encode("utf-8"))
        assert n_bytes < 10_000, (
            f"{d.doc_id} body is {n_bytes} bytes; expected well under the "
            "~49KB Vietnamese trace-safe threshold"
        )


# ---------------------------------------------------------------------------
# n_docs edge cases
# ---------------------------------------------------------------------------


def test_custom_n_docs_is_respected_and_still_covers_every_trap():
    c = Corpus.generate(seed=42, n_docs=30)
    assert len(c.docs) == 30
    tags = {t for d in c.docs for t in d.tags}
    for trap in TRAP_CLASSES:
        assert trap in tags


def test_n_docs_too_small_raises_value_error():
    with pytest.raises(ValueError):
        Corpus.generate(seed=42, n_docs=3)
