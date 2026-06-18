#!/usr/bin/env python3
"""CLI: DBpedia schema export, instance subgraph filtering, MASCHInE ontology merge."""

from __future__ import annotations

import argparse
from pathlib import Path

from _dbpedia.entities import load_entity_uris
from _dbpedia.graph_build import report_seed_coverage, seeds_with_subject_triple
from _dbpedia.nt_stream import extract_subgraph_k_hops, iter_nt_iri_triples
from _dbpedia.iri_nt_materialize import copy_ontology_nt, stream_rdf_sources_to_iri_nt
from _dbpedia.schema_rdfs import ontology_to_protograph_schema_nt
from _dbpedia.sparql_fetch import concat_nt_files, fetch_construct_batches
from _dbpedia.v1_collect import collect_dbpedia_resource_uris, write_sorted_uris


def _cmd_export_schema(args: argparse.Namespace) -> None:
    n = ontology_to_protograph_schema_nt(args.ontology, args.output)
    print(f"Wrote {n} rdfs:domain|range|subClassOf triples -> {args.output}")


def _cmd_filter_instance(args: argparse.Namespace) -> None:
    uris = load_entity_uris(*args.entities)
    if not uris:
        raise SystemExit("No URIs loaded from --entities files.")
    n = extract_subgraph_k_hops(
        args.mapping_based,
        args.output,
        seed_uris=uris,
        hops=args.hops,
    )
    print(f"Wrote {n} triples (hops={args.hops}, seeds={len(uris)}) -> {args.output}")


def _cmd_filter_types(args: argparse.Namespace) -> None:
    uris = load_entity_uris(*args.entities)
    if not uris:
        raise SystemExit("No URIs loaded from --entities files.")
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as out:
        for s, p, o in iter_nt_iri_triples(args.instance_types):
            if p != rdf_type or s not in uris:
                continue
            out.write(f"<{s}> <{p}> <{o}> .\n")
            n += 1
    print(f"Wrote {n} rdf:type triples (subject in seed set) -> {args.output}")


def _cmd_merge_maschine_ontology(args: argparse.Namespace) -> None:
    """Concatenate schema + types (+ optional extras), dedupe identical lines (streaming)."""
    seen: set[str] = set()
    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip() or line.startswith("#"):
                        continue
                    if line in seen:
                        continue
                    seen.add(line)
                    out.write(line + "\n")
                    n += 1
    print(f"Wrote {n} unique lines -> {args.output}")


def _cmd_fetch_sparql(args: argparse.Namespace) -> None:
    uris = sorted(load_entity_uris(*args.entities))
    if not uris:
        raise SystemExit("No URIs loaded from --entities files.")
    n = fetch_construct_batches(
        uris,
        args.output,
        endpoint=args.endpoint,
        batch_size=args.batch_size,
        delay_s=args.delay,
    )
    print(f"Wrote {n} unique IRI triples -> {args.output}")


def _cmd_list_v1_entities(args: argparse.Namespace) -> None:
    seeds = collect_dbpedia_resource_uris(args.v1_root)
    if args.output is not None:
        write_sorted_uris(args.output, seeds)
        print(f"Wrote {len(seeds)} unique resource URIs -> {args.output}")
        return
    for u in sorted(seeds):
        print(u)


def _cmd_build_v1_instance(args: argparse.Namespace) -> None:
    if args.entities_file is not None:
        seeds = load_entity_uris(args.entities_file)
    else:
        seeds = collect_dbpedia_resource_uris(args.v1_root)
        if args.cache_entities is not None:
            write_sorted_uris(args.cache_entities, seeds)
            print(f"Cached {len(seeds)} URIs -> {args.cache_entities}")
    if not seeds:
        raise SystemExit("No DBpedia resource URIs found.")

    has_dump = args.mapping_based is not None
    has_sp = bool(args.sparql_only)
    if has_dump == has_sp:
        raise SystemExit("Specify exactly one of: --mapping-based <dump> OR --sparql-only.")

    if has_sp:
        lim = args.max_sparql_entities
        if lim > 0 and len(seeds) > lim and not args.force_sparql:
            raise SystemExit(
                f"{len(seeds)} seeds exceed --max-sparql-entities={lim}. "
                "Use a mapping dump with --mapping-based, or pass --force-sparql."
            )
        n = fetch_construct_batches(
            sorted(seeds),
            args.output,
            endpoint=args.endpoint,
            batch_size=args.batch_size,
            delay_s=args.delay,
        )
        print(f"SPARQL-only: wrote {n} triples -> {args.output}")
    else:
        n = extract_subgraph_k_hops(
            args.mapping_based,
            args.output,
            seed_uris=seeds,
            hops=args.hops,
        )
        print(f"Dump extract: wrote {n} triples (hops={args.hops}) -> {args.output}")
        if args.augment_with_sparql:
            as_subj = seeds_with_subject_triple(args.output, seeds)
            orphans = sorted(seeds - as_subj)
            if not orphans:
                print("Augment: all seeds already have ≥1 outgoing IRI triple; skip SPARQL.")
            else:
                cap = args.max_augment_entities
                if cap > 0 and len(orphans) > cap:
                    print(
                        f"Augment: {len(orphans)} seeds lack outgoing triples; "
                        f"SPARQL-fetching first {cap} (--max-augment-entities).",
                        flush=True,
                    )
                    orphans = orphans[:cap]
                aug = args.output.parent / f"{args.output.name}.sparql_augment.tmp"
                m = fetch_construct_batches(
                    orphans,
                    aug,
                    endpoint=args.endpoint,
                    batch_size=min(args.batch_size, 15),
                    delay_s=args.delay,
                )
                merged = args.output.parent / f"{args.output.name}.merged.tmp"
                concat_nt_files([args.output, aug], merged)
                merged.replace(args.output)
                aug.unlink(missing_ok=True)
                print(f"Augment: merged {m} SPARQL triples for {len(orphans)} seeds.")

    n_seeds, n_touch, n_sub = report_seed_coverage(args.output, seeds)
    print(
        f"Seed coverage: total={n_seeds}, appear_in_graph={n_touch}, appear_as_subject={n_sub}",
        flush=True,
    )
    if n_touch < n_seeds:
        raise SystemExit(
            f"{n_seeds - n_touch} seeds never appear as subject or object in {args.output}. "
            "Increase --hops or use a richer dump / --augment-with-sparql."
        )
    if n_sub < n_seeds:
        print(
            f"Note: {n_seeds - n_sub} seeds have no outgoing IRI triple (jRDF2Vec roots may be weak). "
            "Consider --augment-with-sparql or a dump with more predicates.",
            flush=True,
        )


def _cmd_materialize_dbpedia_graph_dir(args: argparse.Namespace) -> None:
    """
    ``ontology.nt`` = copy of parsed ontology NT; ``graph.nt`` = streaming merge of
    mapping-based object triples + instance-types (IRI–IRI–IRI only), for large Turtle.bz2 dumps.
    """
    d = args.dir
    onto_in = args.ontology_src or d / "ontology--DEV_type=parsed.nt"
    map_in = args.mapping_based or d / "mappingbased-objects_lang=en.ttl.bz2"
    types_in = args.instance_types or d / "instance-types_lang=en_specific.ttl.bz2"
    onto_out = args.ontology_out or d / "ontology.nt"
    graph_out = args.graph_out or d / "graph.nt"

    copy_ontology_nt(onto_in, onto_out)
    print(f"Copied ontology -> {onto_out}")

    n = stream_rdf_sources_to_iri_nt([map_in, types_in], graph_out, append=False)
    print(f"Wrote {n} IRI triples (mapping + instance-types) -> {graph_out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build DBpedia N-Triples for RDF2Vec protographs, instance walks, and MASCHInE init.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("export-schema", help="Extract rdfs:domain/range/subClassOf from ontology OWL/TTL/NT.")
    s1.add_argument("--ontology", type=Path, required=True, help="DBpedia ontology file (e.g. dbpedia.owl).")
    s1.add_argument("-o", "--output", type=Path, required=True, help="Output .nt for uv run protograph --schema")
    s1.set_defaults(func=_cmd_export_schema)

    s2 = sub.add_parser(
        "filter-instance",
        help="Extract subgraph from mapping-based (or similar) .nt/.bz2 touching seed URIs.",
    )
    s2.add_argument(
        "--entities",
        type=Path,
        nargs="+",
        required=True,
        help="Text files with one resource URI per line (positives, negatives, train, …).",
    )
    s2.add_argument(
        "--mapping-based",
        type=Path,
        required=True,
        help="Large DBpedia object-properties dump (.nt or .nt.bz2).",
    )
    s2.add_argument(
        "--hops",
        type=int,
        default=1,
        help="Undirected hop radius from seeds (default: 1).",
    )
    s2.add_argument("-o", "--output", type=Path, required=True, help="Output instance subgraph .nt")
    s2.set_defaults(func=_cmd_filter_instance)

    s3 = sub.add_parser("filter-types", help="Keep rdf:type rows whose subject is in the seed URI set.")
    s3.add_argument("--entities", type=Path, nargs="+", required=True)
    s3.add_argument(
        "--instance-types",
        type=Path,
        required=True,
        help="DBpedia instance-types .nt or .bz2 (IRI objects only rows are kept).",
    )
    s3.add_argument("-o", "--output", type=Path, required=True)
    s3.set_defaults(func=_cmd_filter_types)

    s4 = sub.add_parser(
        "merge-maschine-ontology",
        help="Merge schema .nt + types .nt (dedupe lines) for train_word2vec --ontology.",
    )
    s4.add_argument("inputs", type=Path, nargs="+", help="Order: typically schema first, then types.")
    s4.add_argument("-o", "--output", type=Path, required=True)
    s4.set_defaults(func=_cmd_merge_maschine_ontology)

    s5 = sub.add_parser(
        "fetch-sparql",
        help="Build instance .nt via batched SPARQL CONSTRUCT (no local dump required).",
    )
    s5.add_argument("--entities", type=Path, nargs="+", required=True)
    s5.add_argument(
        "--endpoint",
        default="https://dbpedia.org/sparql",
        help="SPARQL endpoint URL (default: DBpedia).",
    )
    s5.add_argument("--batch-size", type=int, default=25, metavar="N")
    s5.add_argument("--delay", type=float, default=0.5, help="Seconds between batch requests.")
    s5.add_argument("-o", "--output", type=Path, required=True)
    s5.set_defaults(func=_cmd_fetch_sparql)

    s6 = sub.add_parser(
        "list-v1-entities",
        help="Print (or write) all http://dbpedia.org/resource/... IRIs found under v1/dbpedia/**/*.txt.",
    )
    s6.add_argument(
        "--v1-root",
        type=Path,
        default=Path("v1/dbpedia"),
        help="Root directory to scan (default: v1/dbpedia).",
    )
    s6.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="If set, write sorted URIs one per line instead of printing to stdout.",
    )
    s6.set_defaults(func=_cmd_list_v1_entities)

    s7 = sub.add_parser(
        "build-v1-instance",
        help="Build dbpedia.nt: all v1/dbpedia resource URIs plus k-hop context from a dump, or SPARQL-only.",
    )
    s7.add_argument(
        "--v1-root",
        type=Path,
        default=Path("v1/dbpedia"),
        help="Scan this tree when --entities-file is omitted.",
    )
    s7.add_argument(
        "--entities-file",
        type=Path,
        default=None,
        help="Optional: newline-separated URIs (skip scanning v1-root).",
    )
    s7.add_argument(
        "--cache-entities",
        type=Path,
        default=None,
        help="When scanning v1-root, also write sorted URIs to this path.",
    )
    s7.add_argument(
        "--mapping-based",
        type=Path,
        default=None,
        help="DBpedia mapping-based object-properties .nt or .nt.bz2 (recommended for full v1).",
    )
    s7.add_argument(
        "--sparql-only",
        action="store_true",
        help="Fetch all triples ?s ?p ?o with IRI objects via SPARQL (slow for large v1).",
    )
    s7.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Undirected hop radius from seeds when using --mapping-based (default: 2).",
    )
    s7.add_argument(
        "--augment-with-sparql",
        action="store_true",
        help="After dump extract, SPARQL-fetch outgoing triples for seeds missing as subject.",
    )
    s7.add_argument(
        "--max-augment-entities",
        type=int,
        default=50_000,
        metavar="N",
        help="Cap orphan seeds sent to SPARQL augment (0 = no cap).",
    )
    s7.add_argument(
        "--endpoint",
        default="https://dbpedia.org/sparql",
        help="SPARQL endpoint for --sparql-only / --augment-with-sparql.",
    )
    s7.add_argument("--batch-size", type=int, default=25, metavar="N")
    s7.add_argument("--delay", type=float, default=0.5, help="Seconds between SPARQL batches.")
    s7.add_argument(
        "--max-sparql-entities",
        type=int,
        default=5_000,
        metavar="N",
        help="Abort --sparql-only if more than N seeds unless --force-sparql (0 = no limit).",
    )
    s7.add_argument(
        "--force-sparql",
        action="store_true",
        help="Allow --sparql-only even when above --max-sparql-entities.",
    )
    s7.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("v1/dbpedia/dbpedia.nt"),
        help="Output instance graph (default: v1/dbpedia/dbpedia.nt).",
    )
    s7.set_defaults(func=_cmd_build_v1_instance)

    s8 = sub.add_parser(
        "materialize-dbpedia-graph-dir",
        help="From dbpedia_graph-style dumps: copy ontology NT + build graph.nt (Turtle.bz2 → IRI NT via pyoxigraph).",
    )
    s8.add_argument(
        "--dir",
        type=Path,
        default=Path("dbpedia_graph"),
        help="Folder containing default filenames (default: dbpedia_graph).",
    )
    s8.add_argument(
        "--ontology-src",
        type=Path,
        default=None,
        help="Source ontology .nt (default: <dir>/ontology--DEV_type=parsed.nt).",
    )
    s8.add_argument(
        "--mapping-based",
        type=Path,
        default=None,
        help="Mapping-based Turtle dump (default: <dir>/mappingbased-objects_lang=en.ttl.bz2).",
    )
    s8.add_argument(
        "--instance-types",
        type=Path,
        default=None,
        help="Instance-types Turtle dump (default: <dir>/instance-types_lang=en_specific.ttl.bz2).",
    )
    s8.add_argument(
        "--ontology-out",
        type=Path,
        default=None,
        help="Output ontology copy (default: <dir>/ontology.nt).",
    )
    s8.add_argument(
        "--graph-out",
        type=Path,
        default=None,
        help="Merged IRI graph output (default: <dir>/graph.nt).",
    )
    s8.set_defaults(func=_cmd_materialize_dbpedia_graph_dir)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
