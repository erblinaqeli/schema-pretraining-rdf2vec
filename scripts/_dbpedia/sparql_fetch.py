"""Optional bounded SPARQL CONSTRUCT to build a small instance graph without local dumps."""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rdflib import Graph, URIRef
from tqdm.auto import tqdm


def _construct_query(uris: list[str]) -> str:
    body = " ".join(f"<{u}>" for u in uris)
    return f"""CONSTRUCT {{ ?s ?p ?o }}
WHERE {{
  VALUES ?s {{ {body} }}
  ?s ?p ?o .
  FILTER (isIRI(?o))
}}"""


def _parse_sparql_construct(raw: str) -> Graph:
    g = Graph()
    for fmt in ("turtle", "nt", "xml"):
        try:
            g.parse(data=raw, format=fmt)
            if len(g) > 0:
                return g
        except Exception:
            continue
    # empty graph is ok
    return g


def fetch_construct_batches(
    uris: list[str],
    output_nt: Path,
    *,
    endpoint: str,
    batch_size: int,
    delay_s: float,
) -> int:
    """
    POST SPARQL CONSTRUCT in batches; write deduplicated IRI-object triples as N-Triples.
    """
    output_nt.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    batches = [uris[i : i + batch_size] for i in range(0, len(uris), batch_size)]
    with output_nt.open("w", encoding="utf-8") as out:
        for batch in tqdm(batches, desc="SPARQL CONSTRUCT batches", unit="batch"):
            q = _construct_query(batch)
            data = urllib.parse.urlencode({"query": q}).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=data,
                method="POST",
                headers={
                    "Accept": "text/turtle, application/n-triples, text/plain",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                raise SystemExit(f"SPARQL HTTP error: {e.code} {e.reason}\n{e.read()[:500]!r}") from e
            except urllib.error.URLError as e:
                raise SystemExit(f"SPARQL network error: {e}") from e
            g = _parse_sparql_construct(raw)
            for s, p, o in g:
                if not isinstance(s, URIRef) or not isinstance(p, URIRef) or not isinstance(o, URIRef):
                    continue
                t = (str(s), str(p), str(o))
                if t in seen:
                    continue
                seen.add(t)
                out.write(f"<{t[0]}> <{t[1]}> <{t[2]}> .\n")
            if delay_s > 0:
                time.sleep(delay_s)
    return len(seen)


def concat_nt_files(parts: list[Path], dest: Path) -> int:
    """Append binary contents of ``parts`` into ``dest`` (creates/overwrites ``dest``). Returns total bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("wb") as w:
        for p in parts:
            chunk = p.read_bytes()
            w.write(chunk)
            total += len(chunk)
    return total
