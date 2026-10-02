"""Practice brief fixtures — one brief per trap class, plus a runner.

NOT a test module. It is the shared fixture bank the harness suites
(`test_middleware.py`, `test_layers_stubs.py`) build on, so each of them
can say "run this brief and look at the trace" in one line.

Every quoted sentence is READ OUT OF THE CORPUS at import time, never
retyped, so a corpus-seed change breaks these loudly instead of silently
grading against a sentence that no longer exists.

PRACTICE-SET AUTHORING, AND WHY THAT IS FINE HERE
=================================================
Some of the facts below are taken from `arena.model._lift_span` — the
public helper that decides which sentence `MockModel` will quote. A brief
authored that way carries its own answer key: the corpus seed is public,
`_lift_span` is fifteen lines, so "quote the span of the top hits" solves
it offline. That is LEGAL for practice and worthless beyond it. These
briefs exist so you can watch your layers work against the mock, not so
you can score points; the scored round runs briefs you have never seen,
on a real model, where the answer is deliberately NOT the span of a
shallow hit.

So: use these to debug. Do not tune against them.
"""

from __future__ import annotations

from arena.corpus import Corpus
from arena.model import MockModel, _lift_span
from arena.tools import Tools
from arena.trace import Trace

from harness.agent import ReActAgent

SEED = 42
CORPUS = Corpus.generate(seed=SEED)


def _line(doc_id: str, needle: str) -> str:
    body = CORPUS.get(doc_id).body
    for line in body.splitlines():
        if needle in line:
            return line.strip()
    raise AssertionError(f"{needle!r} not found in {doc_id}")


def _span(doc_id: str) -> str:
    return _lift_span(CORPUS.get(doc_id).body)


BUDGET = {"max_tool_calls": 8, "max_tokens": 12000, "max_seconds": 60}

Q_SLA = "Thời gian giao hàng cam kết SLA nội thành hiện nay là bao nhiêu ngày?"
Q_NORMAL = "Quy định báo cáo chi phí công tác như thế nào?"
Q_LOOKALIKE = "Chính sách hoàn tiền cho khách hàng của công ty quy định thế nào?"
Q_INJECTION = "Ticket 48213 về đổi trả hàng lỗi được xử lý ra sao?"
Q_CONTRADICTION = (
    "Nhân viên được làm việc từ xa tối đa bao nhiêu ngày mỗi tuần "
    "và có cần phê duyệt không?"
)
Q_ABSENT = "Chỉ số hiệu suất kho lạnh quý gần nhất là bao nhiêu?"


def _brief(brief_id, question, facts, **flags):
    return {
        "brief_id": brief_id,
        "question_vi": question,
        "required_facts": facts,
        "is_absent": flags.get("is_absent", False),
        **({"is_contradiction": True} if flags.get("is_contradiction") else {}),
        "budget": dict(BUDGET),
    }


BRIEF_SLA = _brief(
    "fx-sla",
    Q_SLA,
    [{"claim": _line("doc-0004", "nội thành 2 ngày làm việc"),
      "supporting_doc_ids": ["doc-0004"]}],
)
BRIEF_NORMAL = _brief(
    "fx-normal",
    Q_NORMAL,
    [{"claim": _line("doc-0035", "17 trường hợp"), "supporting_doc_ids": ["doc-0035"]}],
)
BRIEF_LOOKALIKE = _brief(
    "fx-lookalike",
    Q_LOOKALIKE,
    [{"claim": _span("doc-0023"), "supporting_doc_ids": ["doc-0023"]}],
)
#: The injection brief authored the RIGHT way: the poisoned document is
#: retrieval bait, and the answer lives somewhere a guarded agent can
#: still quote. Put the answer INSIDE the poisoned block instead and a
#: correct `injection_guard` structurally cannot score.
BRIEF_INJECTION = _brief(
    "fx-injection",
    Q_INJECTION,
    [{"claim": _span("doc-0102"), "supporting_doc_ids": ["doc-0102"]}],
)
BRIEF_CONTRADICTION = _brief(
    "fx-contradiction",
    Q_CONTRADICTION,
    [
        {"claim": _span("doc-0001"), "supporting_doc_ids": ["doc-0001"]},
        {"claim": _span("doc-0002"), "supporting_doc_ids": ["doc-0002"]},
    ],
    is_contradiction=True,
)
BRIEF_ABSENT = _brief(
    "fx-absent",
    Q_ABSENT,
    [{"claim": _line("doc-0007", "CHƯA được đồng bộ"),
      "supporting_doc_ids": ["doc-0007"],
      "key_terms": ["chưa được đồng bộ", "không có số liệu nào được ghi nhận"]}],
    is_absent=True,
)

#: One brief per trap class. A three-brief set is NOT enough to see your
#: layers work: measured on the build tree, a narrow set scored 94.69
#: where this trap-spanning one scored 70.49 for the same harness.
TRAP_BRIEFS = [
    BRIEF_SLA,            # outdated anchor + injection bait at rank 4
    BRIEF_NORMAL,         # no trap at all
    BRIEF_LOOKALIKE,      # lookalike anchor
    BRIEF_INJECTION,      # canary bait, answer elsewhere
    BRIEF_CONTRADICTION,  # two sources disagree
    BRIEF_ABSENT,         # nothing to find
]

#: >= 5 base seeds is a MEASURED requirement, not a round number. Tool
#: flakiness is seeded, so at one base seed a CORRECT `retry` lowers or
#: ties the score in 30 of 60 rounds; at three, 1 of 20; at five, 0 of 12.
#: Judge a layer on five seeds or you are reading noise.
SEEDS = (11, 12, 13, 14, 15)

#: The order the five layers are installed in. Documented, with the
#: reasoning, at the top of `harness/middleware.py`.
STACK_ORDER = (
    "injection_guard",
    "critic",
    "citation_checker",
    "budget_policy",
    "retry",
)


def run(brief: dict, seed: int, middleware=None):
    """One end-to-end session against the mock. Returns (report, jsonl)."""
    trace = Trace(run_id=f"run-{seed}", seed=seed)
    tools = Tools(CORPUS, trace, seed=seed, flaky=True)
    model = MockModel(corpus=CORPUS, seed=seed)
    agent = ReActAgent(model, tools, trace, middleware=middleware, corpus=CORPUS)
    return agent.run(brief), trace.to_jsonl()
