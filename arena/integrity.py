"""Integrity: is the code that produced this score the code we froze?

Everything else in this lab assumes `arena/` is the instructor's. The
scorer's provenance rule, the runner's three clauses, the `k` clamp, the
event reconciliation — all of it is enforced by code a student could
replace. This module is the check that the code was not replaced, and it
runs on BOTH sides of a scored run because the two failure modes are
different:

* **before** — the tree on disk is wrong (a student's `arena/` was copied
  into the sandbox, an instructor edited a frozen file, a partial restore).
* **after** — the tree was right at import time and something changed it
  during the run: an on-disk write, or a monkeypatch that never touched
  the disk at all (`arena.scorer._supports = lambda *a: True`).

A disk hash cannot see the second one and an in-memory fingerprint cannot
see the first, so both are here and `IntegrityGuard` runs both.

WHAT IS FROZEN, AND WHY THE LIST IS PATHS AND NOT HASHES
========================================================

`FROZEN_FILES` names five paths. It does NOT carry their digests: those
are read off the files at manifest-build time and written to
`phases/frozen.manifest.json`. A hash list typed into source goes stale
the moment someone legitimately re-freezes a file — which happened once
already in this build, when `arena/scorer.py` was unfrozen to add the
synthesis rules and re-frozen at a new digest. Build the manifest, commit
the manifest, verify against the manifest.

`PINNED_FILES` is the rest of `arena/`. It is verified with the same
machinery but reported separately, because "the instructor edited
`runner.py`" and "a student edited `scorer.py`" are not the same event.
In a grading SANDBOX both are fatal: the sandbox is a pristine restore,
so anything that differs there differs because something moved it.

THE IN-MEMORY FINGERPRINT
=========================

`module_fingerprint` digests a module's attributes: functions and classes
by the CONTENT of their code objects (recursively, so a nested lambda
counts), simple constants by value, and anything else by type. Rebinding
`Trace.emit`, replacing `_supports`, deleting a constant or adding a
shim all change the digest. Live caches do not: a container whose
elements are not simple values is digested by TYPE only, which is what
keeps `scorer._BODY_CACHE` (a `WeakKeyDictionary` that fills up during
scoring) from reading as tampering.

WHAT THIS IS NOT
================

It is not a sandbox. A harness that can run native code can defeat any
Python-level check, this one included. The containment that does not
depend on the harness's good behaviour is architectural and lives in
`scripts/grade_submissions.py`: the answer key is never in the student's
process, and the score is computed in a different process from the one
that ran their code.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: The five files whose contents this whole lab's measurements assume.
#: Order is the order they are reported in.
FROZEN_FILES = (
    "arena/trace.py",
    "arena/corpus.py",
    "arena/tools.py",
    "arena/model.py",
    "arena/scorer.py",
)

#: The rest of `arena/`. Instructor-owned, not student-owned, and equally
#: fatal inside a grading sandbox — see the module docstring.
PINNED_FILES = (
    "arena/__init__.py",
    "arena/briefs.py",
    "arena/integrity.py",
    "arena/runner.py",
)

#: Import names matching `FROZEN_FILES`, for the in-memory check.
FROZEN_MODULES = (
    "arena.trace",
    "arena.corpus",
    "arena.tools",
    "arena.model",
    "arena.scorer",
)

#: Import names matching `PINNED_FILES` that are worth fingerprinting.
#: `arena.__init__` is empty and `arena.integrity` is this module.
PINNED_MODULES = (
    "arena.briefs",
    "arena.runner",
)

MANIFEST_SCHEMA = "arena-frozen-manifest/1"

#: Where `scripts/phase.py init` writes the manifest. Under `phases/`
#: because the manifest is phase infrastructure: it is what lets a
#: grading run assert that the `arena/` it just restored is the one the
#: numbers were measured against.
MANIFEST_RELATIVE_PATH = "phases/frozen.manifest.json"

_HASH_READ_CHUNK = 1 << 20


class IntegrityError(RuntimeError):
    """Raised when a tree or a live module set does not match.

    Carries `report` so a caller that wants the structured detail does
    not have to parse the message.
    """

    def __init__(self, message: str, report: "IntegrityReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Hashing files
# ---------------------------------------------------------------------------


def file_digests(path) -> dict:
    """`{"sha256": ..., "md5": ..., "size": ...}` for one file.

    Both algorithms, one read. `sha256` is the one integrity decisions are
    made on; `md5` is kept because every frozen-file number in this
    build's ledger and in `scripts/verify.py` is an MD5, and a manifest
    that cannot be checked against those by eye is a manifest nobody
    checks.
    """
    path = Path(path)
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            sha.update(chunk)
            md5.update(chunk)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "size": size}


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    category: str
    sha256: str
    md5: str
    size: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "category": self.category,
            "sha256": self.sha256,
            "md5": self.md5,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestEntry":
        return cls(
            path=str(data["path"]),
            category=str(data.get("category", "frozen")),
            sha256=str(data["sha256"]),
            md5=str(data.get("md5", "")),
            size=int(data.get("size", 0)),
        )


@dataclass(frozen=True)
class Manifest:
    """What `arena/` looked like when someone said "this is the build"."""

    entries: tuple
    created: str = ""
    note: str = ""
    schema: str = MANIFEST_SCHEMA

    def by_path(self) -> dict:
        return {entry.path: entry for entry in self.entries}

    def paths(self, categories=("frozen", "pinned")) -> list:
        wanted = set(categories)
        return [e.path for e in self.entries if e.category in wanted]

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "created": self.created,
            "note": self.note,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        if not isinstance(data, dict):
            raise IntegrityError("manifest is not a JSON object")
        schema = str(data.get("schema", ""))
        if schema != MANIFEST_SCHEMA:
            raise IntegrityError(
                f"manifest schema {schema!r} != {MANIFEST_SCHEMA!r}; "
                "rebuild it with `python3 scripts/phase.py init --write-manifest`"
            )
        raw = data.get("entries")
        if not isinstance(raw, list) or not raw:
            raise IntegrityError("manifest carries no entries")
        return cls(
            entries=tuple(ManifestEntry.from_dict(item) for item in raw),
            created=str(data.get("created", "")),
            note=str(data.get("note", "")),
            schema=schema,
        )


def lab_root() -> Path:
    """The lab tree this module was imported from."""
    return Path(__file__).resolve().parent.parent


def build_manifest(root=None, *, note: str = "") -> Manifest:
    """Hash the files ON DISK RIGHT NOW and return a manifest.

    This is the "read the hashes from the files" half of the contract.
    Nothing in this module hardcodes a digest, so re-freezing a file is
    one command rather than a hunt through source for stale constants.
    """
    root = Path(root) if root is not None else lab_root()
    entries = []
    for category, group in (("frozen", FROZEN_FILES), ("pinned", PINNED_FILES)):
        for relative in group:
            path = root / relative
            if not path.is_file():
                raise IntegrityError(
                    f"cannot build a manifest: {relative} is missing from {root}"
                )
            digests = file_digests(path)
            entries.append(
                ManifestEntry(
                    path=relative,
                    category=category,
                    sha256=digests["sha256"],
                    md5=digests["md5"],
                    size=digests["size"],
                )
            )
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Manifest(entries=tuple(entries), created=created, note=note)


def write_manifest(path, manifest: Manifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path=None) -> Manifest:
    path = Path(path) if path is not None else lab_root() / MANIFEST_RELATIVE_PATH
    if not Path(path).is_file():
        raise IntegrityError(
            f"no frozen manifest at {path}. Build one with "
            "`python3 scripts/phase.py init --write-manifest`."
        )
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError(f"cannot read the manifest at {path}: {exc}") from exc
    return Manifest.from_dict(data)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

#: Every way a check can come out wrong. `MODIFIED` is the one that
#: matters; the others exist so the message can name what actually
#: happened instead of "mismatch".
PROBLEM_KINDS = (
    "MODIFIED",
    "MISSING",
    "UNREADABLE",
    "REBOUND",
    "REMOVED",
    "ADDED",
)


@dataclass(frozen=True)
class Problem:
    path: str
    kind: str
    category: str = "frozen"
    expected: str = ""
    actual: str = ""
    detail: str = ""

    def line(self) -> str:
        head = f"{self.kind} [{self.category}] {self.path}"
        if self.expected and self.actual:
            return f"{head}: {self.actual} != {self.expected}"
        if self.detail:
            return f"{head}: {self.detail}"
        return head

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "kind": self.kind,
            "category": self.category,
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class IntegrityReport:
    """The answer to one integrity question, with the evidence attached."""

    where: str
    checked: int
    problems: tuple = ()
    when: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems

    def message(self) -> str:
        if self.ok:
            return f"{self.where}: {self.checked} item(s) verified, no differences"
        lines = [
            f"INTEGRITY FAILURE ({self.where}): "
            f"{len(self.problems)} of {self.checked} item(s) do not match.",
            "Refusing to score. The offending path(s):",
        ]
        lines.extend(f"  - {problem.line()}" for problem in self.problems)
        return "\n".join(lines)

    def raise_if_bad(self) -> "IntegrityReport":
        if not self.ok:
            raise IntegrityError(self.message(), self)
        return self

    def to_dict(self) -> dict:
        return {
            "where": self.where,
            "ok": self.ok,
            "checked": self.checked,
            "when": self.when,
            "problems": [problem.to_dict() for problem in self.problems],
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def merge_reports(reports, where: str) -> IntegrityReport:
    """One report out of several, keeping every problem."""
    reports = list(reports)
    problems = tuple(p for report in reports for p in report.problems)
    return IntegrityReport(
        where=where,
        checked=sum(report.checked for report in reports),
        problems=problems,
        when=_now(),
    )


# ---------------------------------------------------------------------------
# The disk check
# ---------------------------------------------------------------------------


def verify_tree(
    root=None,
    manifest: Manifest | None = None,
    *,
    categories=("frozen", "pinned"),
    where: str = "",
) -> IntegrityReport:
    """Compare a tree on disk against a manifest. Never raises for a
    mismatch — it REPORTS one, so a caller can decide whether a mismatch
    is fatal (a grading sandbox: yes) or a warning (an instructor's own
    working tree mid-edit: their call).
    """
    root = Path(root) if root is not None else lab_root()
    manifest = manifest if manifest is not None else load_manifest()
    wanted = set(categories)
    problems = []
    checked = 0
    for entry in manifest.entries:
        if entry.category not in wanted:
            continue
        checked += 1
        path = root / entry.path
        if not path.is_file():
            problems.append(
                Problem(entry.path, "MISSING", entry.category, detail=f"not in {root}")
            )
            continue
        try:
            digests = file_digests(path)
        except Exception as exc:
            problems.append(
                Problem(
                    entry.path,
                    "UNREADABLE",
                    entry.category,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if digests["sha256"] != entry.sha256:
            problems.append(
                Problem(
                    entry.path,
                    "MODIFIED",
                    entry.category,
                    expected=f"sha256:{entry.sha256[:16]}",
                    actual=f"sha256:{digests['sha256'][:16]}",
                    detail=f"{digests['size']} bytes on disk, {entry.size} expected",
                )
            )
    return IntegrityReport(
        where=where or f"frozen files under {root}",
        checked=checked,
        problems=tuple(problems),
        when=_now(),
    )


def require_tree(root=None, manifest=None, **kwargs) -> IntegrityReport:
    """`verify_tree` that raises `IntegrityError` on any difference."""
    return verify_tree(root, manifest, **kwargs).raise_if_bad()


# ---------------------------------------------------------------------------
# The in-memory check
# ---------------------------------------------------------------------------

#: Module attributes never fingerprinted. Import machinery, not code.
_SKIPPED_ATTRIBUTES = frozenset(
    {
        "__builtins__",
        "__cached__",
        "__file__",
        "__loader__",
        "__spec__",
        "__path__",
        "__doc__",
        "__name__",
        "__package__",
        "__warningregistry__",
        "annotations",  # `from __future__ import annotations`
    }
)

#: Values digested by VALUE. Everything else is digested by type, which
#: is what keeps a live cache from reading as tampering.
_SIMPLE_TYPES = (str, bytes, int, float, bool, type(None))


def _is_simple(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, _SIMPLE_TYPES):
        if isinstance(value, (tuple, frozenset)):
            return all(_is_simple(item) for item in value)
        return isinstance(value, _SIMPLE_TYPES)
    return True


def _canon_repr(value) -> str:
    """`repr` with set iteration order removed.

    NOT cosmetic. `repr(frozenset({"a", "b"}))` depends on
    `PYTHONHASHSEED`, and frozensets appear in `co_consts` all over the
    frozen modules (every `x in {"a", "b"}` compiles to one). Plain
    `repr` therefore fingerprints the same module differently in two
    processes — measured, before this function existed: 6 of 7 modules
    unstable across two hash seeds.
    """
    if isinstance(value, (set, frozenset)):
        inner = ", ".join(sorted(_canon_repr(item) for item in value))
        return f"{type(value).__name__}({{{inner}}})"
    if isinstance(value, tuple):
        return "(" + ", ".join(_canon_repr(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_canon_repr(item) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(
            f"{_canon_repr(k)}: {_canon_repr(v)}" for k, v in value.items()
        )
        return "{" + ", ".join(items) + "}"
    return repr(value)


def _feed_code(code, digest) -> None:
    """Digest a code object by CONTENT, recursing into nested code.

    `repr(code.co_consts)` cannot be used directly: a nested function's
    code object reprs with its memory address, so the same module would
    fingerprint differently in two processes.
    """
    digest.update(b"code\x00")
    digest.update(str(code.co_name).encode("utf-8", "replace"))
    digest.update(b"\x00")
    digest.update(code.co_code)
    digest.update(b"\x00")
    digest.update(_canon_repr(code.co_names).encode("utf-8", "replace"))
    digest.update(_canon_repr(code.co_varnames).encode("utf-8", "replace"))
    digest.update(_canon_repr(code.co_freevars).encode("utf-8", "replace"))
    digest.update(_canon_repr(code.co_cellvars).encode("utf-8", "replace"))
    digest.update(str(code.co_argcount).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _feed_code(const, digest)
        else:
            digest.update(_canon_repr(const).encode("utf-8", "replace"))
        digest.update(b"\x00")


def _feed_value(value, digest, depth: int = 0) -> None:
    if depth > 6:  # pragma: no cover - defensive
        digest.update(b"deep")
        return
    unwrapped = getattr(value, "__wrapped__", None)
    if unwrapped is not None and callable(value):
        digest.update(b"wrapped\x00")
        _feed_value(unwrapped, digest, depth + 1)
        return
    if isinstance(value, (staticmethod, classmethod)):
        digest.update(b"bound\x00")
        _feed_value(value.__func__, digest, depth + 1)
        return
    if isinstance(value, property):
        digest.update(b"property\x00")
        for part in (value.fget, value.fset, value.fdel):
            _feed_value(part, digest, depth + 1)
        return
    if isinstance(value, types.FunctionType):
        _feed_code(value.__code__, digest)
        digest.update(_canon_repr(value.__defaults__).encode("utf-8", "replace"))
        return
    if isinstance(value, types.BuiltinFunctionType):
        digest.update(f"builtin:{getattr(value, '__qualname__', '?')}".encode())
        return
    if isinstance(value, type):
        digest.update(f"class:{value.__qualname__}\x00".encode("utf-8", "replace"))
        digest.update(
            _canon_repr([base.__qualname__ for base in value.__bases__]).encode()
        )
        for name in sorted(vars(value)):
            if name in ("__dict__", "__weakref__", "__doc__"):
                continue
            digest.update(name.encode("utf-8", "replace"))
            digest.update(b"=")
            _feed_value(vars(value)[name], digest, depth + 1)
            digest.update(b";")
        return
    if isinstance(value, types.ModuleType):
        digest.update(
            f"module:{getattr(value, '__name__', '?')}:"
            f"{getattr(value, '__file__', '?')}".encode("utf-8", "replace")
        )
        return
    if _is_simple(value):
        digest.update(f"value:{type(value).__name__}:".encode())
        digest.update(_canon_repr(value).encode("utf-8", "replace"))
        return
    if isinstance(value, (list, tuple, set, frozenset)) and all(
        _is_simple(item) for item in value
    ):
        digest.update(f"seq:{type(value).__name__}:".encode())
        digest.update(_canon_repr(value).encode("utf-8", "replace"))
        return
    if isinstance(value, dict) and all(
        _is_simple(k) and _is_simple(v) for k, v in value.items()
    ):
        digest.update(b"map:")
        digest.update(_canon_repr(value).encode("utf-8", "replace"))
        return
    # A live cache, a compiled regex, a weak dictionary. TYPE only: its
    # contents change during an honest run and none of them decide a score.
    digest.update(f"opaque:{type(value).__name__}".encode())


def attribute_fingerprints(module) -> dict:
    """`{attribute name: digest}` for one module."""
    out = {}
    for name, value in list(vars(module).items()):
        if name in _SKIPPED_ATTRIBUTES:
            continue
        digest = hashlib.sha256()
        try:
            _feed_value(value, digest)
        except Exception as exc:  # pragma: no cover - exotic objects
            digest.update(f"unfingerprintable:{type(exc).__name__}".encode())
        out[name] = digest.hexdigest()
    return out


def module_fingerprint(module) -> str:
    """One digest over a whole module's attributes."""
    digest = hashlib.sha256()
    for name, value in sorted(attribute_fingerprints(module).items()):
        digest.update(name.encode("utf-8", "replace"))
        digest.update(b"=")
        digest.update(value.encode())
        digest.update(b";")
    return digest.hexdigest()


def snapshot_modules(names=None, *, import_missing: bool = True) -> dict:
    """Fingerprint the frozen modules AS THEY ARE IN THIS PROCESS.

    Take this before importing any student code. Anything the student's
    import time does to `arena` — a decorator, a patched method, a
    replaced constant — lands between this snapshot and the check.
    """
    names = tuple(names) if names is not None else FROZEN_MODULES + PINNED_MODULES
    snapshot = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            if not import_missing:
                continue
            module = __import__(name, fromlist=["_"])
        snapshot[name] = attribute_fingerprints(module)
    return snapshot


def verify_modules(snapshot: dict, *, where: str = "live arena modules") -> IntegrityReport:
    """Compare the live modules against a `snapshot_modules` result."""
    problems = []
    checked = 0
    for name, expected in sorted(snapshot.items()):
        module = sys.modules.get(name)
        if module is None:
            checked += 1
            problems.append(
                Problem(name, "REMOVED", "module", detail="module no longer imported")
            )
            continue
        actual = attribute_fingerprints(module)
        for attribute, digest in sorted(expected.items()):
            checked += 1
            if attribute not in actual:
                problems.append(
                    Problem(
                        f"{name}.{attribute}",
                        "REMOVED",
                        "module",
                        detail="attribute deleted during the run",
                    )
                )
            elif actual[attribute] != digest:
                problems.append(
                    Problem(
                        f"{name}.{attribute}",
                        "REBOUND",
                        "module",
                        expected=digest[:16],
                        actual=actual[attribute][:16],
                        detail="attribute rebound or mutated during the run",
                    )
                )
        for attribute in sorted(set(actual) - set(expected)):
            checked += 1
            problems.append(
                Problem(
                    f"{name}.{attribute}",
                    "ADDED",
                    "module",
                    detail="attribute added during the run",
                )
            )
    return IntegrityReport(
        where=where, checked=checked, problems=tuple(problems), when=_now()
    )


# ---------------------------------------------------------------------------
# Both halves, in the order a scored run needs them
# ---------------------------------------------------------------------------


@dataclass
class IntegrityGuard:
    """Verify before, verify after. The whole contract in one object.

        guard = IntegrityGuard.arm(root, manifest)   # raises if the tree is wrong
        ...run the student's harness...
        guard.check_after().raise_if_bad()           # raises if anything moved

    `arm` is deliberately a constructor that RAISES: a caller who forgets
    to look at the return value still cannot score a tampered tree.
    """

    root: Path
    manifest: Manifest
    categories: tuple = ("frozen", "pinned")
    before: IntegrityReport | None = None
    after: IntegrityReport | None = None
    module_snapshot: dict = field(default_factory=dict)

    @classmethod
    def arm(
        cls,
        root=None,
        manifest=None,
        *,
        categories=("frozen", "pinned"),
        modules=None,
        where: str = "before the run",
    ) -> "IntegrityGuard":
        root = Path(root) if root is not None else lab_root()
        manifest = manifest if manifest is not None else load_manifest()
        before = verify_tree(
            root, manifest, categories=categories, where=where
        ).raise_if_bad()
        return cls(
            root=root,
            manifest=manifest,
            categories=tuple(categories),
            before=before,
            module_snapshot=snapshot_modules(modules),
        )

    def check_after(self, *, where: str = "after the run") -> IntegrityReport:
        """Disk AND memory, merged into one report. Does not raise."""
        disk = verify_tree(
            self.root,
            self.manifest,
            categories=self.categories,
            where=f"{where} (on disk)",
        )
        live = verify_modules(self.module_snapshot, where=f"{where} (in memory)")
        self.after = merge_reports((disk, live), where=where)
        return self.after

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "manifest_created": self.manifest.created,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
        }


__all__ = [
    "FROZEN_FILES",
    "FROZEN_MODULES",
    "MANIFEST_RELATIVE_PATH",
    "MANIFEST_SCHEMA",
    "PINNED_FILES",
    "PINNED_MODULES",
    "PROBLEM_KINDS",
    "IntegrityError",
    "IntegrityGuard",
    "IntegrityReport",
    "Manifest",
    "ManifestEntry",
    "Problem",
    "attribute_fingerprints",
    "build_manifest",
    "file_digests",
    "lab_root",
    "load_manifest",
    "merge_reports",
    "module_fingerprint",
    "require_tree",
    "snapshot_modules",
    "verify_modules",
    "verify_tree",
    "write_manifest",
]
