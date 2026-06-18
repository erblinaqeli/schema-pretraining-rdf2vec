"""Config helpers, runtime metrics, output layout, and shared I/O utilities."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

TRIPLE_LINE = re.compile(
    r"^<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*$",
    re.UNICODE,
)

WalkRole = Literal["pretrain_proto", "instance", "no_pretrain"]

# Default RNG seed for reproducible walk generation when YAML omits random_seed.
DEFAULT_WALK_SEED = 42


def repo_root() -> Path:
    """Repository root: the nearest ancestor directory containing pyproject.toml.

    Found by walking up from this file, so the answer does not depend on how deep
    the importing module sits. Scripts that import this keep resolving the same
    v1/, walks/, output/, and jar paths even after being relocated into subfolders.
    Falls back to the parent of scripts/ if pyproject.toml is not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[1]


def output_root() -> Path:
    return repo_root() / "output"


def cache_root() -> Path:
    return output_root() / "_cache"


def index_path() -> Path:
    return output_root() / "index" / "experiments.jsonl"


def walk_cache_key(
    w: dict[str, Any],
    *,
    tm: str,
    role: WalkRole,
) -> str:
    """Canonical walk cache key: {mode}_d{depth}_wpe{wpe}[_tf{token_format}][_s{seed}][_cov].

    The jRDF2Vec JAR cannot be seeded (its walk RNG is an unseeded ThreadLocalRandom),
    so jar-mode keys carry no seed suffix: those walks are only reproducible via this cache.
    ``cov`` marks protograph pretrain walks generated with --ensure-triple-coverage.
    """
    mode = str(w.get("mode", "jrdf2vec-duplicate-free"))
    depth = int(w["depth"])
    wpe = int(w["walks_per_entity"])
    parts = [mode, f"d{depth}", f"wpe{wpe}"]
    tf = w.get("token_format")
    if tf is not None and str(tf) != "angled":
        parts.append(f"tf_{tf}")
    if mode != "jrdf2vec-jar":
        seed: int | None = None
        if w.get("random_seed") is not None:
            seed = int(w["random_seed"])
        elif role == "pretrain_proto" and tm in ("p1", "p2"):
            seed = DEFAULT_WALK_SEED
        elif role in ("instance", "no_pretrain"):
            seed = DEFAULT_WALK_SEED
        if seed is not None:
            parts.append(f"s{seed}")
    if role == "pretrain_proto" and mode == "jrdf2vec-duplicate-free":
        parts.append("cov")
    return "_".join(parts)


def build_run_dir(
    *,
    dataset: str,
    slug: str,
    timestamp: str,
    tc: str | None = None,
    mode: str | None = None,
    family: str | None = None,
) -> Path:
    """Build experiment run directory (new layout).

    Synthetic with family:
      P1/P2: output/synthetic/protograph/{slug}/[{tc}/]{timestamp}/{mode}/
      Vanilla: output/synthetic/vanilla/{slug}/[{tc}/]{timestamp}/ (no mode leaf)

    Other datasets (family omitted):
      P1/P2: output/{dataset}/{slug}/[{tc}/]{timestamp}/{mode}/
      Vanilla: output/{dataset}/{slug}/[{tc}/]{timestamp}/ (no mode leaf)
    """
    parts: list[str] = ["output", dataset]
    if family is not None:
        parts.append(family)
    parts.append(slug)
    if tc is not None:
        parts.append(tc)
    parts.append(timestamp)
    if mode is not None and mode not in ("no_pretrain", "vanilla"):
        parts.append(mode)
    return repo_root().joinpath(*parts)


def protograph_cache_dir(*, dataset: str, tc: str | None) -> Path:
    if dataset == "synthetic" and tc is not None:
        return cache_root() / "protograph" / "synthetic" / tc
    return cache_root() / "protograph" / dataset


def walk_cache_dir(
    *,
    dataset: str,
    tc: str | None,
    role: str,
    walk_key: str,
    tm: str | None = None,
) -> Path:
    base = cache_root() / "walks" / dataset
    if dataset == "synthetic" and tc is not None:
        base = base / tc
    if role == "pretrain" and tm in ("p1", "p2"):
        return base / "pretrain" / tm / walk_key
    return base / role / walk_key


def walk_cache_file(
    *,
    dataset: str,
    tc: str | None,
    role: str,
    walk_key: str,
    tm: str | None = None,
) -> Path:
    return walk_cache_dir(
        dataset=dataset, tc=tc, role=role, walk_key=walk_key, tm=tm
    ) / "walks.txt"


def check_cache_stale(meta_path: Path, input_path: Path) -> bool:
    """Return True if cache entry is stale or missing metadata."""
    if not meta_path.is_file():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    stored = meta.get("input_mtime")
    if stored is None:
        return False
    if not input_path.is_file():
        return True
    return float(stored) != input_path.stat().st_mtime


def write_cache_meta(meta_path: Path, payload: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifact_refs(run_dir: Path, refs: dict[str, str], *, dry_run: bool) -> None:
    if dry_run:
        print(f"  artifact_refs → {run_dir / 'artifact_refs.json'}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifact_refs.json").write_text(
        json.dumps(refs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_runtime_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "runtime_metrics.json"
    if not path.is_file():
        return {"stages": {}, "total": 0.0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"stages": {}, "total": 0.0}
    stages = data.get("stages", {})
    if not isinstance(stages, dict):
        stages = {}
    total = data.get("total", sum(float(v) for v in stages.values()))
    return {"stages": stages, "total": float(total)}


def append_index_entry(entry: dict[str, Any]) -> None:
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


class RuntimeMetrics:
    """Accumulate stage durations in seconds."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._started_at = datetime.now(timezone.utc).isoformat()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.stages[name] = self.stages.get(name, 0.0) + elapsed

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def total_seconds(self) -> float:
        return sum(self.stages.values())

    def to_dict(self) -> dict[str, object]:
        total = self.total_seconds()
        return {
            "unit": "seconds",
            "started_at": self._started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "stages": dict(sorted(self.stages.items())),
            "total": round(total, 3),
        }


def format_runtime_metrics_text(data: dict[str, object]) -> str:
    stages = data.get("stages")
    if not isinstance(stages, dict):
        stages = {}
    lines = [
        "Runtime metrics (wall-clock)",
        f"Started:  {data.get('started_at', '')}",
        f"Finished: {data.get('finished_at', '')}",
        f"Unit:     {data.get('unit', 'seconds')}",
        "",
    ]
    unit = str(data.get("unit", "seconds"))
    for name, value in stages.items():
        lines.append(f"  {name}: {float(value):.3f} {unit}")
    lines.append("")
    lines.append(f"  total: {float(data.get('total', 0.0)):.3f} {unit}")
    return "\n".join(lines) + "\n"


def write_runtime_metrics(out_dir: Path, metrics: RuntimeMetrics) -> tuple[Path, Path]:
    """Write ``runtime_metrics.json`` and ``runtime_metrics.txt`` under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = metrics.to_dict()
    json_path = out_dir / "runtime_metrics.json"
    txt_path = out_dir / "runtime_metrics.txt"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    txt_path.write_text(format_runtime_metrics_text(data), encoding="utf-8")
    return json_path, txt_path


def load_runtime_stages(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    stages = raw.get("stages", raw)
    if not isinstance(stages, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in stages.items():
        if key == "total":
            continue
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def merge_runtime_metrics(out_dir: Path, pipeline: RuntimeMetrics) -> tuple[Path, Path]:
    """
    Merge *pipeline* timings with an existing ``runtime_metrics.json`` in *out_dir*
    (e.g. written by ``train_word2vec.py``), then rewrite both JSON and text files.
    """
    existing_path = out_dir / "runtime_metrics.json"
    merged = RuntimeMetrics()
    merged._started_at = pipeline._started_at
    for name, seconds in load_runtime_stages(existing_path).items():
        merged.stages[name] = seconds
    for name, seconds in pipeline.stages.items():
        merged.stages[name] = merged.stages.get(name, 0.0) + seconds
    return write_runtime_metrics(out_dir, merged)
