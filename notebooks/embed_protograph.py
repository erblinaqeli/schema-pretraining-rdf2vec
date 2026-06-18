"""Run jRDF2Vec on protograph .nt files (p1, p2, and extended variants)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAR_PATH = ROOT / "jars" / "jrdf2vec-1.3-SNAPSHOT_seed.jar"

DEFAULT_NUM_WALKS = 200
DEFAULT_DEPTH = 3
DEFAULT_DIMENSIONS = 200

PROTO_VARIANTS = (
    "p1",
    "p2",
    "p2_inv",
    "p2_hier",
    "p2_inv_hier",
    "p2_depth2",
    "p2_depth2_inv_hier",
    "p2_depth3",
    "p3_joint",
    "p3_joint_inv_hier",
)


def parse_tc_arg(tc_arg: str) -> list[int]:
    """Parse TCs: '1', '1-12', '1,3,5', '1 2 3', or 'all'."""
    s = str(tc_arg).strip().lower()
    if s == "all":
        return list(range(1, 13))
    if "-" in s:
        start, end = s.split("-", 1)
        return list(range(int(start.strip()), int(end.strip()) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(p) for p in s.split()]


def parse_proto_arg(proto_arg: str) -> list[str]:
    s = str(proto_arg).strip().lower()
    if s == "all":
        return list(PROTO_VARIANTS)
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def embed_one_tc(tc: int, proto_version: str):
    """Run jRDF2Vec for one TC and one protograph variant."""
    tc_str = f"tc{tc:02d}"
    proto_path = (
        ROOT
        / "training_output"
        / "synthetic_ontology"
        / tc_str
        / "protographs"
        / f"protograph_{proto_version}.nt"
    )

    if not proto_path.exists():
        print(f"[WARN] Protograph not found for {tc_str}: {proto_path}")
        print(f"       Run: python scripts_new/build_protographs_variants.py {tc}")
        return

    out_dir = (
        ROOT
        / "training_output"
        / "synthetic_ontology"
        / tc_str
        / "protographs"
        / proto_version
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not JAR_PATH.exists():
        raise FileNotFoundError(f"jRDF2Vec jar not found: {JAR_PATH}")

    cmd = [
        "java", "-jar", str(JAR_PATH),
        "-graph", str(proto_path),
        "-walkDirectory", str(out_dir),
        "-numberOfWalks", str(DEFAULT_NUM_WALKS),
        "-depth", str(DEFAULT_DEPTH),
        "-dimension", str(DEFAULT_DIMENSIONS),
    ]

    print(f"[INFO] Running jRDF2Vec for {tc_str} ({proto_version})...")
    subprocess.run(cmd, check=True)
    print(f"[OK] Finished {tc_str} ({proto_version}) -> {out_dir}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train RDF2Vec embeddings on protograph variants via jRDF2Vec.",
    )
    parser.add_argument(
        "tcs",
        nargs="*",
        default=["all"],
        help='TCs to run: 1, 1-12, 1,3,5, or all (default: all)',
    )
    parser.add_argument(
        "--proto",
        default="p1",
        help='Variant(s): p1, p2, p2_inv_hier, comma list, or "all" (default: p1)',
    )
    args = parser.parse_args(argv)

    protos = parse_proto_arg(args.proto)
    unknown = sorted(set(protos) - set(PROTO_VARIANTS))
    if unknown:
        parser.error(f"Unknown variant(s): {', '.join(unknown)}")

    tc_nums: list[int] = []
    for tc_arg in args.tcs:
        tc_nums.extend(parse_tc_arg(tc_arg))
    tc_nums = sorted(set(tc_nums))

    for tc in tc_nums:
        for proto in protos:
            embed_one_tc(tc, proto)


if __name__ == "__main__":
    main(sys.argv[1:])