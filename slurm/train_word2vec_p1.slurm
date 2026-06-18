#!/bin/bash
#SBATCH --job-name=rdf2vec
#SBATCH --partition=cpu
#SBATCH --output=/pfs/work9/workspace/scratch/ma_eqeli-thesis/Knowledge-Graphs/slurm/%x-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_eqeli-thesis/Knowledge-Graphs/slurm/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

# Repository root (must match paths in #SBATCH lines above).
REPO="/pfs/work9/workspace/scratch/ma_eqeli-thesis/Knowledge-Graphs"
cd "${REPO}" || exit 1

mkdir -p "${REPO}/slurm"

module purge
module load devel/miniforge

export PATH="${REPO}/bin:${PATH}"
export UV_CACHE_DIR="${REPO}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${REPO}/.uv-python"

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

which uv || exit 2
uv --version || exit 3

# --- Hardcoded inputs ---
# Put (or symlink) walks_p1.txt and walks_instance.txt here before submitting, e.g. from
# output/<run>/ after scripts/e2e_p1.sh steps 1–3.
TC="tc12"
RUN_INPUTS="${REPO}/slurm/input/tc12_p1_walks"
ONTOLOGY_NT="${REPO}/v1/synthetic_ontology/${TC}/synthetic_ontology/ontology.nt"
WALKS_P1="${RUN_INPUTS}/walks_p1.txt"
WALKS_INSTANCE="${RUN_INPUTS}/walks_instance.txt"

# All Word2Vec outputs for this job land under slurm/.
OUT="${REPO}/slurm/train_word2vec_p1/job-${SLURM_JOB_ID}"
CHECKPOINT="${OUT}/rdf2vec_final.pt"
mkdir -p "${OUT}"

uv run python scripts/_word2vec.py \
  --mode p1 \
  --pretrain-walks "${WALKS_P1}" \
  --instance-walks "${WALKS_INSTANCE}" \
  --ontology "${ONTOLOGY_NT}" \
  --out-dir "${OUT}" \
  --architecture skipgram \
  --dim 100 \
  --window 5 \
  --pretrain-epochs 5 \
  --finetune-epochs 5 \
  --loss-every-steps-pretrain 1 \
  --loss-every-steps-finetune 100 \
  -o "${CHECKPOINT}"
