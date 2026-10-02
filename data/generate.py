#!/usr/bin/env python3
"""CLI: materialise the Agent Arena hostile corpus to `data/corpus/`.

    python data/generate.py --seed 42

Writes one JSON file per `Doc` plus a `manifest.json`. This is a
convenience artifact so students can browse a fixed corpus
(`cat data/corpus/doc-0001.json`) without running any Python — it is
NOT the harness's runtime source of truth. The harness calls
`Corpus.generate(seed)` directly at runtime, and by construction (see
`arena/corpus.py`'s determinism guarantee: one `random.Random(seed)`
instance, fixed-order pools, no `hash()`/set-iteration dependence)
that call reproduces byte-identical `Doc` bodies to what this script
writes — that equivalence is exactly what `tests/test_corpus.py`'s
determinism tests check.

Running this script writes files to disk under `data/corpus/`. That is
NOT a git commit — this repo has zero commits by design and no git
command is ever run here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

from arena.corpus import Corpus  # noqa: E402  (path setup must run first)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialise the Agent Arena hostile corpus to disk."
    )
    parser.add_argument("--seed", type=int, default=42, help="generation seed (default: 42)")
    parser.add_argument("--n-docs", type=int, default=120, help="number of documents (default: 120)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_LAB_ROOT / "data" / "corpus",
        help="output directory (default: data/corpus)",
    )
    args = parser.parse_args(argv)

    corpus = Corpus.generate(seed=args.seed, n_docs=args.n_docs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for doc in corpus.docs:
        path = args.out_dir / f"{doc.doc_id}.json"
        payload = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "body": doc.body,
            "tags": list(doc.tags),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "seed": args.seed,
        "n_docs": len(corpus.docs),
        "doc_ids": [d.doc_id for d in corpus.docs],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(corpus.docs)} documents (seed={args.seed}) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
