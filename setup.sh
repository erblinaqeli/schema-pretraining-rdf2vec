#!/usr/bin/env bash
# Lay out the artifacts a fresh clone needs and check prerequisites.
# Automates what it can (data unzip, jar placement) and reports what is missing.
# Full walkthrough: docs/SETUP.md
#
# Usage:
#   ./setup.sh           # locate data + jar, run prerequisite check
#   ./setup.sh --smoke   # the above, then a synthetic dry-run of the pipeline
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

JAR="jrdf2vec-1.3-SNAPSHOT_seed.jar"
SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
err()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }

missing=0

echo "== 1. Environment =="
if command -v uv >/dev/null 2>&1; then
  ok "uv found"
  if uv run python -c "import torch, gensim, rdflib, sklearn" >/dev/null 2>&1; then
    ok "python deps import (torch, gensim, rdflib, sklearn)"
  else
    err "deps not importable — run: uv sync"; missing=1
  fi
else
  err "uv not found — install uv, then run: uv sync"; missing=1
fi

echo "== 2. Benchmark data (v1/) =="
if [ -f "v1/synthetic_ontology/tc01/synthetic_ontology/graph.nt" ]; then
  ok "v1/ present"
else
  zip=""
  [ -f "v1.zip" ] && zip="v1.zip"
  [ -z "$zip" ] && [ -f "v1 (1).zip" ] && zip="v1 (1).zip"
  if [ -n "$zip" ]; then
    warn "v1/ missing — unzipping $zip"
    unzip -n -q "$zip" && ok "unzipped $zip" || { err "unzip failed"; missing=1; }
  else
    err "v1/ missing and no v1 zip found — see docs/SETUP.md §2"; missing=1
  fi
fi
# extra synthetic test cases extract straight into v1/synthetic_ontology/
if [ -f "new_tcs.zip" ] && [ -d "v1/synthetic_ontology" ]; then
  unzip -n -q new_tcs.zip -d v1/synthetic_ontology/ && ok "extra TCs from new_tcs.zip" || warn "new_tcs.zip extract failed"
fi

echo "== 3. jRDF2Vec jar =="
if [ -f "jars/$JAR" ]; then
  ok "jars/$JAR present"
elif [ -f "$JAR" ]; then
  mkdir -p jars && mv "$JAR" "jars/$JAR" && ok "moved $JAR -> jars/"
else
  err "jar missing — place a custom seeded jRDF2Vec build at jars/$JAR (see docs/SETUP.md §3)"; missing=1
fi

echo "== 4. DBpedia graph (optional; only for --dataset dbpedia) =="
if [ -f "dbpedia_graph/graph.nt" ]; then
  ok "dbpedia_graph/graph.nt present"
else
  warn "not materialized — run: uv run python scripts/train.py --prepare-dbpedia materialize --dir dbpedia_graph (see docs/SETUP.md §4)"
fi

echo
if [ "$missing" -ne 0 ]; then
  err "Setup incomplete — resolve the ✗ items above, then re-run ./setup.sh"
  exit 1
fi
ok "Prerequisites satisfied."

if [ "$SMOKE" -eq 1 ]; then
  echo
  echo "== Smoke test (synthetic dry-run) =="
  uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2 --dry-run
fi

echo
echo "Next:  uv run python scripts/train.py --dataset synthetic --tc tc12 --train-mode p2"
