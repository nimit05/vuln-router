#!/usr/bin/env python3
"""A1 + A2 -- CWE nodes from MITRE's own catalogue, with hierarchy edges.

Phase A step 1: "Create CWE nodes from the MITRE descriptions, encoded. Add
parent/child edges from the CWE hierarchy."

Source is `cwec_latest.xml` (the official catalogue), not the hand-seeded map in
src/vulnrouter/cwe.py -- that map exists so the code runs without a download and
is deliberately short, because a wrong ChildOf edge pools two unrelated classes
and the pooling is invisible downstream. With the real catalogue there is no
reason to guess.

Only **ChildOf** relations from **view 1000** (the Research view) are kept.
PeerOf, CanPrecede and CanAlsoBe are associations, not generalisations; treating
them as hierarchy would mean a CWE inherits statistics from a class it is not a
specialisation of.

Encoder is `all-MiniLM-L6-v2`, matching GraphRouter's `utils.get_embedding`.
The paper says "a mid-size PLM"; their code says MiniLM, and the code is what
produced their numbers.
"""
from __future__ import annotations
import argparse, json, os, re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

NS = {"c": "http://cwe.mitre.org/cwe-7"}
ROOT = "CWE-ROOT"


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def text_of(el) -> str:
    """Flatten an element's text, dropping xhtml markup the catalogue embeds."""
    if el is None:
        return ""
    parts = []
    for node in el.iter():
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def parse(xml_path: str, view: str = "1000"):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    nodes, parents = {}, {}
    for w in root.iter():
        if strip_ns(w.tag) != "Weakness":
            continue
        cid = w.get("ID")
        if not cid:
            continue
        key = f"CWE-{int(cid)}"
        name = w.get("Name") or ""
        desc = ext = ""
        for child in w:
            t = strip_ns(child.tag)
            if t == "Description":
                desc = text_of(child)
            elif t == "Extended_Description":
                ext = text_of(child)
            elif t == "Related_Weaknesses":
                for rel in child:
                    if strip_ns(rel.tag) != "Related_Weakness":
                        continue
                    if rel.get("Nature") != "ChildOf":
                        continue
                    if rel.get("View_ID") != view:
                        continue
                    pid = rel.get("CWE_ID")
                    if pid and key not in parents:
                        parents[key] = f"CWE-{int(pid)}"
        # the node feature: name + description, extended text trimmed. The
        # extended text runs to paragraphs for some classes and would dominate
        # a 256-token sentence encoder.
        feature = f"{name}. {desc}"
        if ext:
            feature += " " + ext[:600]
        nodes[key] = dict(id=key, name=name, description=desc, feature=feature)
    return nodes, parents


def chain(cwe: str, parents: dict[str, str], cap: int = 16) -> list[str]:
    out, seen, cur = [], set(), cwe
    while cur and cur not in seen and len(out) < cap:
        out.append(cur)
        seen.add(cur)
        cur = parents.get(cur)
    out.append(ROOT)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    d = os.path.expanduser("~/routing/graphrouter/data")
    ap.add_argument("--xml", default=os.path.join(d, "cwec_v4.20.xml"))
    ap.add_argument("--units", default=os.path.expanduser(
        "~/data/units/units_slices.jsonl"))
    ap.add_argument("--out-parents", default=os.path.join(d, "cwe_parents.json"))
    ap.add_argument("--out-nodes", default=os.path.join(d, "cwe_nodes.json"))
    ap.add_argument("--out-emb", default=os.path.join(d, "cwe_nodes.npz"))
    ap.add_argument("--no-embed", action="store_true")
    a = ap.parse_args()

    nodes, parents = parse(a.xml)
    print(f"catalogue: {len(nodes)} weaknesses, {len(parents)} ChildOf edges (view 1000)")

    # which CWEs do our candidates actually use, and do they resolve?
    used = Counter()
    if os.path.exists(a.units):
        for line in open(a.units):
            if line.strip():
                used[json.loads(line).get("query_cwe")] += 1
    if used:
        print("CWEs in the candidate set:")
        for c, n in used.most_common():
            ch = chain(c, parents)
            here = "ok" if c in nodes else "NOT IN CATALOGUE"
            print(f"  {c:10s} n={n:5d}  {here:16s} chain: {' -> '.join(ch)}")

    # keep every used CWE plus its full ancestor chain: those are the nodes the
    # hierarchy needs in order to pass anything between classes
    keep = set()
    for c in used:
        keep.update(chain(c, parents))
    keep.discard(ROOT)
    keep = {k for k in keep if k in nodes}
    print(f"\nnodes retained (used CWEs + ancestors): {len(keep)}")

    with open(a.out_parents, "w") as fh:
        json.dump({k: v for k, v in parents.items()}, fh, indent=1)
    with open(a.out_nodes, "w") as fh:
        json.dump({k: nodes[k] for k in sorted(keep)}, fh, indent=1)
    print(f"wrote {a.out_parents} and {a.out_nodes}")

    if a.no_embed:
        raise SystemExit(0)
    import numpy as np
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    ids = sorted(keep)
    emb = enc.encode([nodes[i]["feature"] for i in ids], batch_size=32)
    np.savez(a.out_emb, ids=np.array(ids), embeddings=emb)
    print(f"wrote {a.out_emb}: {emb.shape[0]} nodes x {emb.shape[1]} dims")
