"""Extract rdfs:domain, rdfs:range, rdfs:subClassOf from an ontology file for protograph_gen."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS


def ontology_to_protograph_schema_nt(ontology_path: Path, output_nt: Path) -> int:
    """
    Parse ``ontology_path`` with RDFLib (OWL/RDF/XML/Turtle/N-Triples) and write
    only IRI–IRI–IRI triples whose predicate is rdfs:domain, rdfs:range, or
    rdfs:subClassOf. Blank nodes are skipped (protograph_gen expects IRI ends).
    """
    g = Graph()
    g.parse(ontology_path)
    output_nt.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    wanted = (RDFS.domain, RDFS.range, RDFS.subClassOf)
    with output_nt.open("w", encoding="utf-8") as out:
        for p in wanted:
            for s, _p, o in g.triples((None, p, None)):
                if not isinstance(s, URIRef) or not isinstance(o, URIRef):
                    continue
                si, oi = str(s), str(o)
                out.write(f"<{si}> <{_p}> <{oi}> .\n")
                n += 1
    return n
