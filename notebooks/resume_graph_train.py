"""
Graph-only training initialized from protograph classes (Variant B: fresh model with init).

Experiment:
  1) Load protograph KeyedVectors (p1 or p2) from model.kv (class vectors).
  2) Build a NEW Word2Vec model whose vocab is all tokens from graph walks (vanilla/, .gz only).
  3) Use entity2classes.json to initialize instance/entity vectors:
        - For each entity in mapping that is in graph vocab,
          average its class vectors from proto_kv and write into model.wv.
     This gives a graph-only vocab, with shared tokens initialized from protograph.
  4) Train on graph walks only in one continuous train(..., epochs=graph_epochs) so alpha decay is
     correct. A callback saves checkpoints after each epoch (epoch_1, epoch_2, ...).

Outputs (per tc, proto):
  training_output/synthetic_ontology/tcXX/walks/resume_graph/{p1|p2}/
    epoch_0/model.model, epoch_0/model.kv   (init only, before any graph training)
    epoch_1/model.model, epoch_1/model.kv
    epoch_2/model.model, epoch_2/model.kv
    ...
    epochs_metadata.tsv   (graph_epoch, vocab_size, initialized, skipped stats)
    epochs_metadata.json

Usage:
  python scripts_new/resume_graph_train.py --tc 1 --proto p1 --graph-epochs 5
  python scripts_new/resume_graph_train.py --tc 1 --proto p2
  python scripts_new/resume_graph_train.py --tc 1-12 --proto all
  python scripts_new/resume_graph_train.py --tc 1,3,5 --proto p1,p2
"""

import argparse
import gzip
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# Match vanilla: fix RNGs for reproducibility
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts_new") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts_new"))


def parse_tc_arg(tc_arg: str) -> list[int]:
    """Parse --tc: '1' -> [1], '1-12' -> [1..12], '1,3,5' -> [1,3,5]."""
    s = str(tc_arg).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a.strip()), int(b.strip()) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def parse_proto_arg(proto_arg: str) -> list[str]:
    from protograph_variants import parse_proto_arg as _parse_proto_arg

    return _parse_proto_arg(proto_arg)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("resume_graph")


def strip_angle_brackets(x: str) -> str:
    if isinstance(x, str) and x.startswith("<") and x.endswith(">"):
        return x[1:-1]
    return x


# -----------------------------
# Corpus: graph walks (vanilla/ dir, .gz only)
# -----------------------------
class MySentences:
    """Yield one token list per line from a file or directory of .gz walk files. Same tokenization as server."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __iter__(self):
        if self.path.is_dir():
            names = sorted(n for n in os.listdir(self.path) if n.endswith(".gz"))
            for name in names:
                p = self.path / name
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        yield line.split(" ")
        else:
            if str(self.path).endswith(".gz"):
                with gzip.open(self.path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        yield line.split(" ")
            else:
                with self.path.open("rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        yield line.split(" ")


class ReiterableGraphWalks:
    """Re-iterable corpus: each __iter__ yields a fresh pass over graph walks (so train(..., epochs=N) works)."""
    def __init__(self, path: Path):
        self.path = path
    def __iter__(self):
        return iter(MySentences(self.path))


class CheckpointSaver(CallbackAny2Vec):
    """Saves model + KV after each epoch (epoch_1, epoch_2, ...). Appends to metadata_rows; records loss and runtime per epoch."""
    def __init__(self, out_dir: Path, save_each_epoch: bool, metadata_rows: list, init_stats: dict):
        self.out_dir = out_dir
        self.save_each_epoch = save_each_epoch
        self.metadata_rows = metadata_rows
        self.init_stats = init_stats
        self.epoch = 0
        self.cumulative_loss_before = 0.0
        self._epoch_start: float | None = None

    def on_epoch_begin(self, model):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, model):
        self.epoch += 1
        total = model.get_latest_training_loss()
        epoch_loss = total - self.cumulative_loss_before
        self.cumulative_loss_before = total
        runtime_sec = time.perf_counter() - self._epoch_start if self._epoch_start is not None else None
        if self.save_each_epoch:
            epoch_dir = self.out_dir / f"epoch_{self.epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_dir / "model.model"))
            model.wv.save(str(epoch_dir / "model.kv"))
            logger.info(
                "Saved checkpoint epoch %d -> %s (epoch_loss: %.4f, cumulative_loss: %.4f, runtime_sec: %.2f)",
                self.epoch, epoch_dir, epoch_loss, total, runtime_sec or 0,
            )
        else:
            logger.info(
                "Epoch %d done (epoch_loss: %.4f, cumulative_loss: %.4f, runtime_sec: %.2f)",
                self.epoch, epoch_loss, total, runtime_sec or 0,
            )
        self.metadata_rows.append({
            "graph_epoch": self.epoch,
            "loss": epoch_loss,
            "runtime_sec": runtime_sec,
            **self.init_stats,
        })


def init_instance_vectors_from_classes_instance_only(
    model: Word2Vec,
    proto_kv: KeyedVectors,
    inst2classes_raw: dict,
) -> tuple[int, int, int, int]:
    """
    Initialize only tokens that exist in the Word2Vec vocab using class vectors from proto_kv.
    Writes into model.wv.vectors.

    Returns: (initialized, skipped_not_in_vocab, skipped_no_classvec, skipped_bad)
    """
    wv = model.wv
    vectors = wv.vectors

    initialized = 0
    skipped_not_in_vocab = 0
    skipped_no_classvec = 0
    skipped_bad = 0

    for inst_uri, cls_list in inst2classes_raw.items():
        inst_id = strip_angle_brackets(inst_uri)

        if inst_id not in wv.key_to_index:
            skipped_not_in_vocab += 1
            continue

        if isinstance(cls_list, str):
            cls_ids = [strip_angle_brackets(cls_list)]
        elif isinstance(cls_list, list):
            cls_ids = [strip_angle_brackets(x) for x in cls_list]
        else:
            skipped_bad += 1
            continue

        cls_vecs = [proto_kv[c] for c in cls_ids if c in proto_kv]
        if not cls_vecs:
            skipped_no_classvec += 1
            continue

        idx = wv.key_to_index[inst_id]
        vectors[idx] = np.mean(np.vstack(cls_vecs), axis=0).astype(np.float32)
        initialized += 1

    return initialized, skipped_not_in_vocab, skipped_no_classvec, skipped_bad


# -----------------------------
# Paths
# -----------------------------
def get_paths(tc: int, proto: str):
    tc_str = f"tc{tc:02d}"
    proto_kv_path = ROOT / "training_output" / "synthetic_ontology" / tc_str / "protographs" / proto / "model.kv"
    mapping_path = ROOT / "training_output" / "synthetic_ontology" / tc_str / "entity2classes.json"
    graph_walks_path = ROOT / "training_output" / "synthetic_ontology" / tc_str / "walks" / "vanilla"
    out_dir = ROOT / "training_output" / "synthetic_ontology" / tc_str / "walks" / "resume_graph" / proto
    return tc_str, proto_kv_path, mapping_path, graph_walks_path, out_dir


# -----------------------------
# Main
# -----------------------------
def run(tc: int, proto: str, graph_epochs: int, save_each_epoch: bool, number_of_threads: int | None) -> None:
    tc_str, proto_kv_path, mapping_path, graph_walks_path, out_dir = get_paths(tc, proto)

    if not proto_kv_path.exists():
        raise FileNotFoundError(f"Protograph KV not found: {proto_kv_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")
    if not graph_walks_path.exists():
        raise FileNotFoundError(f"Graph walks path not found: {graph_walks_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = number_of_threads if number_of_threads is not None else (os.cpu_count() or 1)

    logger.info("=== Graph-only training with init from protograph: %s %s (workers=%d) ===", tc_str, proto, workers)
    logger.info("Protograph KV:  %s", proto_kv_path)
    logger.info("entity2classes: %s", mapping_path)
    logger.info("Graph walks:    %s", graph_walks_path)
    logger.info("Output dir:     %s", out_dir)

    pipeline_start = time.perf_counter()

    # 1) Load protograph KeyedVectors (class vectors source)
    proto_kv: KeyedVectors = KeyedVectors.load(str(proto_kv_path), mmap=None)
    logger.info("Loaded protograph KV: %d tokens, dim=%d", len(proto_kv), proto_kv.vector_size)

    # 2) Build fresh graph-only vocab from walks
    sentences_for_vocab = MySentences(graph_walks_path)
    model = Word2Vec(
        vector_size=proto_kv.vector_size,
        window=5,
        sg=1,
        hs=0,
        negative=5,
        min_count=1,
        sample=0.0,  # no subsampling, same as train_like_jrdf2vec_server / vanilla
        workers=workers,
        compute_loss=True,
        seed=42,
    )
    t0 = time.perf_counter()
    model.build_vocab(sentences_for_vocab)
    build_vocab_time_sec = time.perf_counter() - t0
    vocab_size = len(model.wv)
    graph_count = model.corpus_count
    logger.info("Vocabulary built. Vocab size: %d, corpus count: %d (build_vocab: %.2fs)", vocab_size, graph_count, build_vocab_time_sec)

    # 3) Initialize instance/entity vectors from classes via entity2classes.json
    inst2classes_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    (
        initialized,
        skipped_not_in_vocab,
        skipped_no_classvec,
        skipped_bad,
    ) = init_instance_vectors_from_classes_instance_only(model, proto_kv, inst2classes_raw)
    init_vectors_time_sec = time.perf_counter() - t0
    logger.info(
        "Init from classes: initialized=%d | skipped_not_in_vocab=%d | skipped_no_classvec=%d | skipped_bad=%d (%.2fs)",
        initialized,
        skipped_not_in_vocab,
        skipped_no_classvec,
        skipped_bad,
        init_vectors_time_sec,
    )

    init_stats = {
        "vocab_size": vocab_size,
        "initialized": initialized,
        "skipped_not_in_vocab": skipped_not_in_vocab,
        "skipped_no_classvec": skipped_no_classvec,
        "skipped_bad": skipped_bad,
    }

    # 4) Save initial checkpoint (epoch_0 = after init, before any graph training; not a training step)
    t0 = time.perf_counter()
    init_dir = out_dir / "epoch_0"
    init_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(init_dir / "model.model"))
    model.wv.save(str(init_dir / "model.kv"))
    epoch0_save_time_sec = time.perf_counter() - t0
    logger.info("Saved init-only checkpoint (epoch_0): %s (%.2fs)", init_dir, epoch0_save_time_sec)

    metadata_rows: list[dict] = []  # callback appends epoch 1..N only; epoch 0 written at save time with no runtime

    # 5) Single continuous train() so alpha decay is correct; callback saves checkpoints after each epoch
    corpus = ReiterableGraphWalks(graph_walks_path)
    checkpoint_cb = CheckpointSaver(out_dir, save_each_epoch, metadata_rows, init_stats)
    logger.info("Training for %d graph epochs (single train() for correct alpha decay) ...", graph_epochs)
    t_train_start = time.perf_counter()
    model.train(
        corpus,
        total_examples=graph_count,
        epochs=graph_epochs,
        callbacks=[checkpoint_cb],
        compute_loss=True,
    )
    total_train_time_sec = time.perf_counter() - t_train_start
    logger.info("Model trained. total_train_time_sec=%.2f", total_train_time_sec)

    total_pipeline_time_sec = time.perf_counter() - pipeline_start

    # Save metadata (epoch 0 = init checkpoint only, no runtime; epoch 1..N = loss + training runtime)
    tsv_path = out_dir / "epochs_metadata.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("graph_epoch\tloss\truntime_sec\tcumulative_runtime_sec\tvocab_size\tinitialized\tskipped_not_in_vocab\tskipped_no_classvec\tskipped_bad\n")
        f.write(f"0\t\t\t\t{vocab_size}\t{initialized}\t{skipped_not_in_vocab}\t{skipped_no_classvec}\t{skipped_bad}\n")
        cum = 0.0
        for r in metadata_rows:
            rt = r.get("runtime_sec")
            if rt is not None:
                cum += rt
            rt_str = f"{rt:.4f}" if rt is not None else ""
            cum_str = f"{cum:.4f}" if rt is not None else ""
            loss_str = "" if r.get("loss") is None else str(r["loss"])
            f.write(
                f"{r['graph_epoch']}\t{loss_str}\t{rt_str}\t{cum_str}\t{r['vocab_size']}\t{r['initialized']}\t"
                f"{r['skipped_not_in_vocab']}\t{r['skipped_no_classvec']}\t{r['skipped_bad']}\n"
            )
    logger.info("Saved %s", tsv_path)

    epochs_json = [{"graph_epoch": 0, "loss": None, "runtime_sec": None, "cumulative_runtime_sec": None, **init_stats}]
    cum = 0.0
    for r in metadata_rows:
        rt = r.get("runtime_sec")
        if rt is not None:
            cum += rt
        epochs_json.append({
            "graph_epoch": r["graph_epoch"],
            "loss": r.get("loss"),
            "runtime_sec": rt,
            "cumulative_runtime_sec": cum if rt is not None else None,
            **{k: r[k] for k in init_stats},
        })

    json_path = out_dir / "epochs_metadata.json"
    json_path.write_text(
        json.dumps(
            {
                "tc": tc_str,
                "proto": proto,
                "graph_walks_path": str(graph_walks_path),
                "graph_corpus_count": graph_count,
                "graph_epochs": graph_epochs,
                "build_vocab_time_sec": build_vocab_time_sec,
                "init_vectors_time_sec": init_vectors_time_sec,
                "epoch0_save_time_sec": epoch0_save_time_sec,
                "total_train_time_sec": total_train_time_sec,
                "total_pipeline_time_sec": total_pipeline_time_sec,
                "vocab_size": vocab_size,
                "initialized": initialized,
                "skipped_not_in_vocab": skipped_not_in_vocab,
                "skipped_no_classvec": skipped_no_classvec,
                "skipped_bad": skipped_bad,
                "epochs": epochs_json,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved %s", json_path)
    logger.info("Done.")


def main():
    ap = argparse.ArgumentParser(
        description="Graph-only Word2Vec initialized from protograph classes via entity2classes.json."
    )
    ap.add_argument(
        "--tc", "-t",
        required=True,
        help="Test case(s): 1, '1-12', '1,3,5'.",
    )
    ap.add_argument(
        "--proto",
        default="all",
        help="Protograph variant(s): p1, p2_inv_hier, comma list, or all (default: all).",
    )
    ap.add_argument("--graph-epochs", type=int, default=5, help="Number of graph training epochs (default: 5)")
    ap.add_argument(
        "--save-each-epoch",
        action="store_true",
        default=True,
        help="Save full model + KV after each graph epoch (default: True). Use --no-save-each-epoch to disable.",
    )
    ap.add_argument(
        "--no-save-each-epoch",
        action="store_false",
        dest="save_each_epoch",
        help="Do not save per-epoch checkpoints.",
    )
    ap.add_argument(
        "--number-of-threads",
        type=int,
        default=None,
        dest="number_of_threads",
        help="Word2Vec workers (default: os.cpu_count() or 1, same as train_vanilla_python). Set to 1 for fair comparison with vanilla if needed.",
    )
    args = ap.parse_args()

    tcs = parse_tc_arg(args.tc)
    protos = parse_proto_arg(args.proto)

    for tc in tcs:
        for proto in protos:
            logger.info(">>> Running tc=%d proto=%s", tc, proto)
            try:
                run(
                    tc=tc,
                    proto=proto,
                    graph_epochs=args.graph_epochs,
                    save_each_epoch=args.save_each_epoch,
                    number_of_threads=args.number_of_threads,
                )
            except FileNotFoundError as e:
                logger.error("Skipping tc=%d proto=%s: %s", tc, proto, e)


if __name__ == "__main__":
    main()