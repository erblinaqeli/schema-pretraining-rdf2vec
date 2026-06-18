#!/usr/bin/env python3
"""
Random walks over an RDF Knowledge Graph in N-Triples (.nt) form.

Each triple <s> <p> <o> . contributes edges between s and o. By default the
entity graph is treated as undirected (common for embedding-style walks); use
--directed to follow only subject → object edges.

Modes
-----
* ``classic`` (default): sample many random walks over an undirected (or
  directed) entity graph; each output token is one node in N-Triples form.
* ``jrdf2vec-duplicate-free``: matches jRDF2Vec
  ``MemoryWalkGenerator.generateDuplicateFreeRandomWalksForEntity`` with
  ``WalkGenerationMode.RANDOM_WALKS_DUPLICATE_FREE`` — forward-only hops
  (subject → object), breadth-first expansion of triple chains, random trimming
  when the walk set exceeds ``--walks-per-entity``, one line per walk as
  ``entity p1 o1 p2 o2 ...`` (see ``--token-format``).

Output: one walk per line, space-separated tokens (see ``--token-format``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, URIRef
from tqdm.auto import tqdm

# Same pattern as convert.py: bare identifiers in angle brackets are accepted.
TRIPLE_LINE = re.compile(r"^\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*(?:#.*)?$")


@dataclass(frozen=True, slots=True)
class Triple:
    """One RDF object triple (subject, predicate, object) with bare inner strings."""

    subject: str
    predicate: str
    object: str


def _detect_input_format(path: Path, explicit: str | None) -> str:
    if explicit:
        explicit = explicit.lower().strip()
        if explicit in {"nt", "ntriples", "n-triples"}:
            return "nt"
        if explicit in {"ttl", "turtle"}:
            return "turtle"
        raise SystemExit(f"Unsupported --input-format: {explicit!r} (use: nt|ttl)")
    suf = path.suffix.lower()
    if suf == ".nt":
        return "nt"
    if suf in {".ttl", ".turtle"}:
        return "turtle"
    raise SystemExit(f"Cannot infer input format from extension {path.suffix!r}. Use --input-format nt|ttl.")


def iter_nt_triples(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                raise ValueError(f"Unrecognized N-Triples line: {line[:200]}")
            yield m.group(1), m.group(2), m.group(3)


def iter_nt_object_triples_lenient(path: Path):
    """IRI–IRI–IRI triples only; skips literals, comments, malformed lines (jRDF2Vec-style)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                continue
            yield m.group(1), m.group(2), m.group(3)


def iter_turtle_object_triples(path: Path):
    """
    IRI–IRI–IRI triples only from a Turtle file.

    Notes:
    - Requires valid Turtle (prefixes, base, etc.); unlike the .nt path we do not support bare
      identifiers like <I_tc01_761> unless they are valid IRIs in Turtle.
    - Literals / blank nodes are skipped to match embedding-walk assumptions.
    """
    g = Graph()
    g.parse(path.as_posix(), format="turtle")
    for s, p, o in tqdm(
        g,
        total=len(g),
        desc="Scanning Turtle (triples)",
        unit="triple",
        dynamic_ncols=True,
        colour="cyan",
        mininterval=0.25,
    ):
        if not isinstance(s, URIRef) or not isinstance(p, URIRef) or not isinstance(o, URIRef):
            continue
        yield str(s), str(p), str(o)


def iter_object_triples(path: Path, *, input_format: str, strict_nt: bool) -> tuple[str, str, str]:
    if input_format == "nt":
        if strict_nt:
            yield from iter_nt_triples(path)
        else:
            yield from iter_nt_object_triples_lenient(path)
        return
    if input_format == "turtle":
        yield from iter_turtle_object_triples(path)
        return
    raise AssertionError(f"Unhandled input_format: {input_format}")


def nt_term(inner: str) -> str:
    return f"<{inner}>"


def build_adjacency(
    path: Path,
    *,
    directed: bool,
    input_format: str,
) -> tuple[dict[str, list[str]], list[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    for s, _p, o in iter_object_triples(path, input_format=input_format, strict_nt=True):
        all_nodes.add(s)
        all_nodes.add(o)
        if directed:
            neighbors[s].add(o)
        else:
            neighbors[s].add(o)
            neighbors[o].add(s)

    # Sort neighbor lists and node order so walks are reproducible for a fixed --seed
    # across machines/Python versions (set iteration order depends on PYTHONHASHSEED).
    adj = {
        n: sorted(neighbors[n])
        for n in tqdm(all_nodes, desc="Materializing neighbor lists", unit="node", dynamic_ncols=True, colour="cyan", mininterval=0.5)
    }
    nodes_sorted = sorted(all_nodes)
    return adj, nodes_sorted


def random_walk(
    adj: dict[str, list[str]],
    *,
    start: str,
    num_nodes: int,
    rng: random.Random,
) -> list[str]:
    """Return a walk of `num_nodes` nodes starting at `start`. Stops early if stuck."""
    if num_nodes < 1:
        return []
    walk = [start]
    cur = start
    for _ in range(num_nodes - 1):
        nbrs = adj.get(cur)
        if not nbrs:
            break
        cur = rng.choice(nbrs)
        walk.append(cur)
    return walk


def build_forward_adjacency(path: Path, *, input_format: str) -> dict[str, list[Triple]]:
    """subject -> outgoing object triples (same as jRDF2Vec ``getObjectTriplesInvolvingSubject``)."""
    adj: dict[str, list[Triple]] = defaultdict(list)
    for s, p, o in iter_object_triples(path, input_format=input_format, strict_nt=False):
        adj[s].append(Triple(s, p, o))
    return {k: v for k, v in adj.items()}


def collect_jrdf2vec_entities(adj: dict[str, list[Triple]]) -> list[str]:
    """All unique subjects ∪ objects of object triples (``MemoryEntitySelector``)."""
    entities: set[str] = set(adj.keys())
    for triples in adj.values():
        for t in triples:
            entities.add(t.object)
    return sorted(entities)


def _rng_for_entity(base_seed: int | None, entity: str) -> random.Random:
    """Stable per-entity RNG (for threaded runs) without relying on ``hash()`` salt."""
    s = base_seed if base_seed is not None else 0
    h = int(hashlib.md5(f"{s}:{entity}".encode()).hexdigest()[:16], 16)
    return random.Random(h)


def generate_duplicate_free_random_walks_for_entity(
    entity: str,
    number_of_walks: int,
    depth: int,
    adj_by_subject: dict[str, list[Triple]],
    rng: random.Random,
) -> list[list[Triple]]:
    """
    Port of jRDF2Vec ``MemoryWalkGenerator.generateDuplicateFreeRandomWalksForEntity``.

    Forward-only: each hop follows ``lastTriple.object`` as the next subject.
    After each depth step, if more than ``number_of_walks`` chains exist, trim
    with ``random.Random.sample`` — same uniform distribution over
    ``number_of_walks``-sized subsets as repeated ``pop(randrange(len))``, but
    not the same RNG stream (per-seed output can differ from a sequential-pop
    implementation).
    """
    walks: list[list[Triple]] = []
    for current_depth in range(depth):
        if current_depth == 0:
            neighbours = adj_by_subject.get(entity)
            if not neighbours:
                return []
            for t in neighbours:
                walks.append([t])
        else:
            new_walks: list[list[Triple]] = []
            for walk in walks:
                last_triple = walk[-1]
                next_iteration = adj_by_subject.get(last_triple.object)
                if next_iteration is not None:
                    for next_step in next_iteration:
                        new_walks.append(walk + [next_step])
                else:
                    new_walks.append(walk)
            walks = new_walks
        if len(walks) > number_of_walks:
            walks = rng.sample(walks, number_of_walks)
    return walks


def walk_triples_to_string(entity: str, triples: list[Triple], *, angled: bool) -> str:
    """Same string layout as jRDF2Vec ``Util.convertToStringWalks(walks, entity, ...)``."""
    parts: list[str] = [entity]
    for t in triples:
        parts.append(t.predicate)
        parts.append(t.object)
    if angled:
        return " ".join(nt_term(p) for p in parts)
    return " ".join(parts)


def run_jrdf2vec_duplicate_free(
    input_graph: Path,
    output_walks: Path,
    *,
    walks_per_entity: int,
    depth: int,
    threads: int,
    seed: int | None,
    token_format: str,
    input_format: str,
    entity_roots: list[str] | None = None,
    ensure_triple_coverage: bool = False,
) -> None:
    adj = build_forward_adjacency(input_graph, input_format=input_format)
    if not adj:
        raise SystemExit("No object triples found; nothing to walk on.")
    all_entities = collect_jrdf2vec_entities(adj)
    if entity_roots is None:
        entities = all_entities
    else:
        present = set(all_entities)
        entities = sorted({e for e in entity_roots if e in present})
        missing = sorted({e for e in entity_roots if e not in present})
        if missing:
            print(
                f"Note: {len(missing)} / {len(entity_roots)} requested entities do not appear in graph; "
                "they will be skipped.",
                flush=True,
            )
        if not entities:
            raise SystemExit("None of the requested entities appear in the graph; nothing to walk.")
    write_jrdf2vec_duplicate_free_walks(
        adj,
        entities,
        output_walks,
        walks_per_entity=walks_per_entity,
        depth=depth,
        threads=threads,
        seed=seed,
        token_format=token_format,
        ensure_triple_coverage=ensure_triple_coverage,
    )


def uncovered_triple_lines(
    adj_by_subject: dict[str, list[Triple]],
    covered: set[Triple],
    *,
    angled: bool,
) -> list[str]:
    """Depth-1 walk lines ``s p o`` for graph triples absent from every kept walk."""
    lines: list[str] = []
    for subj in sorted(adj_by_subject):
        for t in adj_by_subject[subj]:
            if t not in covered:
                lines.append(walk_triples_to_string(subj, [t], angled=angled))
    return lines


def write_jrdf2vec_duplicate_free_walks(
    adj_by_subject: dict[str, list[Triple]],
    entities: list[str],
    output_walks: Path,
    *,
    walks_per_entity: int,
    depth: int,
    threads: int,
    seed: int | None,
    token_format: str,
    ensure_triple_coverage: bool = False,
) -> None:
    angled = token_format == "angled"
    covered: set[Triple] | None = set() if ensure_triple_coverage else None

    def chains_for_entity(entity: str) -> list[list[Triple]]:
        rng = _rng_for_entity(seed, entity)
        return generate_duplicate_free_random_walks_for_entity(
            entity,
            walks_per_entity,
            depth,
            adj_by_subject,
            rng,
        )

    def lines_for_entity(entity: str) -> list[str]:
        triple_chains = chains_for_entity(entity)
        if covered is not None:
            for chain in triple_chains:
                covered.update(chain)
        return [walk_triples_to_string(entity, chain, angled=angled) for chain in triple_chains]

    output_walks.parent.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, threads)
    if n_workers == 1:
        with output_walks.open("w", encoding="utf-8") as out:
            for entity in tqdm(
                entities,
                desc="jRDF2Vec duplicate-free [1 thread]",
                unit="entity",
                dynamic_ncols=True,
                colour="green",
            ):
                for line in lines_for_entity(entity):
                    out.write(line + "\n")
            if covered is not None:
                extra = uncovered_triple_lines(adj_by_subject, covered, angled=angled)
                for line in extra:
                    out.write(line + "\n")
                if extra:
                    print(
                        f"Triple coverage: appended {len(extra)} depth-1 walks for triples "
                        "dropped by per-entity trimming.",
                        flush=True,
                    )
        return

    with output_walks.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=n_workers) as pool:
        # `covered` is only mutated inside worker threads via `lines_for_entity`; set.update
        # is atomic under the GIL, so no extra locking is needed.
        future_to_entity = {pool.submit(lines_for_entity, e): e for e in entities}
        lines_by_entity: dict[str, list[str]] = {}
        for fut in tqdm(
            as_completed(future_to_entity),
            total=len(entities),
            desc=f"jRDF2Vec duplicate-free [{n_workers} threads]",
            unit="entity",
            dynamic_ncols=True,
            colour="green",
        ):
            ent = future_to_entity[fut]
            lines_by_entity[ent] = fut.result()
        for ent in entities:
            for line in lines_by_entity[ent]:
                out.write(line + "\n")
        if covered is not None:
            extra = uncovered_triple_lines(adj_by_subject, covered, angled=angled)
            for line in extra:
                out.write(line + "\n")
            if extra:
                print(
                    f"Triple coverage: appended {len(extra)} depth-1 walks for triples "
                    "dropped by per-entity trimming.",
                    flush=True,
                )


def main() -> None:
    p = argparse.ArgumentParser(description="Random walks on an RDF graph (.nt or .ttl).")
    p.add_argument("input_graph", type=Path, help="Path to input graph file (.nt or .ttl)")
    p.add_argument("output_walks", type=Path, help="Path to output walks file (one walk per line)")
    p.add_argument(
        "--input-format",
        default=None,
        help="Optional: nt or ttl. If omitted, inferred from file extension.",
    )
    p.add_argument(
        "--entities",
        type=Path,
        nargs="+",
        default=None,
        help="Optional: one or more files listing entity URIs to use as walk roots (one URI per line).",
    )
    p.add_argument(
        "--mode",
        choices=("classic", "jrdf2vec-duplicate-free"),
        default="classic",
        help="classic: random node sequences; jrdf2vec-duplicate-free: jRDF2Vec RANDOM_WALKS_DUPLICATE_FREE",
    )
    p.add_argument(
        "--num-walks",
        type=int,
        default=10_000,
        metavar="N",
        help="(classic) Total number of walks to sample (default: 10000)",
    )
    p.add_argument(
        "--walk-length",
        type=int,
        default=40,
        metavar="L",
        help="(classic) Number of nodes per walk (default: 40)",
    )
    p.add_argument(
        "--walks-per-entity",
        type=int,
        default=100,
        metavar="K",
        help="(jrdf2vec-duplicate-free) Max walks kept per entity after each depth trim (default: 100, jRDF2Vec default)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=4,
        metavar="D",
        help="(jrdf2vec-duplicate-free) Number of forward hops (default: 4, jRDF2Vec RDF2Vec default)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="(jrdf2vec-duplicate-free) Worker threads (default: max(1, cpu_count//2))",
    )
    p.add_argument(
        "--token-format",
        choices=("angled", "bare"),
        default="angled",
        help="(jrdf2vec-duplicate-free) angled: <s> <p> <o> ... (fits this repo's train/test); bare: jRDF2Vec file tokens without brackets",
    )
    p.add_argument(
        "--directed",
        action="store_true",
        help="(classic) Follow directed edges s→o only (default: undirected entity graph)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Required for reproducible walks: fixes RNG (classic) and per-entity trim (jrdf2vec); "
        "neighbor lists are sorted so order does not depend on PYTHONHASHSEED",
    )
    p.add_argument(
        "--ensure-triple-coverage",
        action="store_true",
        help="(jrdf2vec-duplicate-free) After trimming, append one depth-1 walk 's p o' for every "
        "graph triple that appears in no kept walk, so all relations/nodes enter the corpus.",
    )
    args = p.parse_args()

    input_format = _detect_input_format(args.input_graph, args.input_format)

    entity_roots: list[str] | None = None
    if args.entities is not None:
        try:
            from _dbpedia.entities import load_entity_uris  # local import to avoid hard coupling
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"Failed to import DBpedia entity loader: {e}") from e
        roots = sorted(load_entity_uris(*args.entities))
        if not roots:
            raise SystemExit("No URIs loaded from --entities files.")
        # Normalize: accept optional angle brackets.
        entity_roots = [r[1:-1] if r.startswith("<") and r.endswith(">") else r for r in roots]

    if args.mode == "jrdf2vec-duplicate-free":
        if args.walks_per_entity < 1:
            raise SystemExit("--walks-per-entity must be at least 1")
        if args.depth < 1:
            raise SystemExit("--depth must be at least 1")
        run_jrdf2vec_duplicate_free(
            args.input_graph,
            args.output_walks,
            walks_per_entity=args.walks_per_entity,
            depth=args.depth,
            threads=args.threads,
            seed=args.seed,
            token_format=args.token_format,
            input_format=input_format,
            entity_roots=entity_roots,
            ensure_triple_coverage=args.ensure_triple_coverage,
        )
        return

    if args.num_walks < 1:
        raise SystemExit("--num-walks must be at least 1")
    if args.walk_length < 1:
        raise SystemExit("--walk-length must be at least 1")

    rng = random.Random(args.seed)
    adj, nodes = build_adjacency(args.input_graph, directed=args.directed, input_format=input_format)
    if not nodes:
        raise SystemExit("No triples found; nothing to walk on.")

    if entity_roots is not None:
        present = set(nodes)
        nodes = [n for n in entity_roots if n in present]
        missing = [n for n in entity_roots if n not in present]
        if missing:
            print(
                f"Note: {len(missing)} / {len(entity_roots)} requested entities do not appear in graph; "
                "they will be skipped.",
                flush=True,
            )
        if not nodes:
            raise SystemExit("None of the requested entities appear in the graph; nothing to walk.")

    args.output_walks.parent.mkdir(parents=True, exist_ok=True)
    mode = "directed" if args.directed else "undirected"
    desc = f"Random walks [{mode}]  target_len={args.walk_length}"
    pbar = tqdm(
        range(args.num_walks),
        desc=desc,
        unit="walk",
        dynamic_ncols=True,
        colour="green",
        smoothing=0.05,
        mininterval=0.2,
        miniters=max(1, args.num_walks // 300),
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    with args.output_walks.open("w", encoding="utf-8") as out:
        for _ in pbar:
            start = rng.choice(nodes)
            walk = random_walk(adj, start=start, num_nodes=args.walk_length, rng=rng)
            line = " ".join(nt_term(n) for n in walk)
            out.write(line + "\n")
            pbar.set_postfix(last_len=len(walk), refresh=False)


if __name__ == "__main__":
    main()
