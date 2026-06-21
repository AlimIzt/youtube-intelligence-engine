"""Lab 2 chunking + Noun-Verb-Noun relation extraction (NLTK RegexpParser).

- `noun_chunks`: groups consecutive nouns with the Lab 2 chunk grammar.
- `extract_relations`: finds Noun-Verb-Noun patterns and formats them as
  verb(subject, object), the Lab 2 relation representation.
"""
from __future__ import annotations

import nltk

from src.utils.nltk_setup import ensure_nltk

# Lab 2 chunk grammar: one or more consecutive nouns form a chunk.
CHUNK_GRAMMAR = r"mychunk: {<NN.*><NN.*>+}"


def _tagged(text: str):
    ensure_nltk()
    return nltk.pos_tag(nltk.word_tokenize(text))


def noun_chunks(text: str) -> list[str]:
    """Extract multi-noun chunks using the Lab 2 RegexpParser grammar."""
    chunker = nltk.RegexpParser(CHUNK_GRAMMAR)
    tree = chunker.parse(_tagged(text))
    chunks = []
    for sub in tree.subtrees(filter=lambda t: t.label() == "mychunk"):
        chunks.append(" ".join(w for w, _ in sub.leaves()))
    return chunks


def extract_relations(text: str) -> list[str]:
    """Extract Noun-Verb-Noun relations as verb(noun1, noun2) (Lab 2)."""
    tags = [(w, t) for w, t in _tagged(text) if t != "DT"]  # drop determiners
    relations = []
    for i in range(len(tags) - 2):
        w1, t1 = tags[i]
        w2, t2 = tags[i + 1]
        w3, t3 = tags[i + 2]
        if t1.startswith("NN") and t2.startswith("VB") and t3.startswith("NN"):
            relations.append(f"{w2}({w1}, {w3})")
    return relations


def relations_for_corpus(texts: list[str], limit: int | None = None) -> list[str]:
    """Run relation extraction over many comments (deduplicated)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in texts[:limit] if limit else texts:
        for rel in extract_relations(str(t)):
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
    return out


def relation_triples(text: str) -> list[tuple[str, str, str]]:
    """Same as extract_relations but as (subject, verb, object) tuples."""
    tags = [(w, t) for w, t in _tagged(text) if t != "DT"]
    triples = []
    for i in range(len(tags) - 2):
        (w1, t1), (w2, t2), (w3, t3) = tags[i], tags[i + 1], tags[i + 2]
        if t1.startswith("NN") and t2.startswith("VB") and t3.startswith("NN"):
            triples.append((w1, w2, w3))
    return triples


def knowledge_graph_dot(texts: list[str], max_edges: int = 30) -> str:
    """Build a Graphviz DOT knowledge graph from Noun-Verb-Noun triples.

    Renders client-side with Streamlit's st.graphviz_chart (no system graphviz).
    """
    from collections import Counter

    triples: Counter = Counter()
    for t in texts:
        for s, v, o in relation_triples(str(t)):
            if s.isalpha() and o.isalpha():
                triples[(s.lower(), v.lower(), o.lower())] += 1

    lines = ["digraph KG {", "  rankdir=LR; node [shape=box, style=rounded];"]
    for (s, v, o), _cnt in triples.most_common(max_edges):
        lines.append(f'  "{s}" -> "{o}" [label="{v}"];')
    lines.append("}")
    return "\n".join(lines)
