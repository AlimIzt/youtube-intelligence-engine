"""Unit tests for Lab 2 Noun-Verb-Noun relation extraction."""

from src.analysis import relations


def test_extract_relations_finds_noun_verb_noun_pattern(monkeypatch):
    """The extractor should format simple Noun-Verb-Noun patterns."""

    monkeypatch.setattr(
        relations,
        "_tagged",
        lambda text: [("Rockstar", "NNP"), ("released", "VBD"), ("trailer", "NN")],
    )

    assert "released(Rockstar, trailer)" in relations.extract_relations("unused")


def test_relation_triples_returns_subject_verb_object_tuple(monkeypatch):
    """The tuple version should preserve subject, verb, object order."""

    monkeypatch.setattr(
        relations,
        "_tagged",
        lambda text: [("Fans", "NNS"), ("love", "VBP"), ("graphics", "NNS")],
    )

    assert ("Fans", "love", "graphics") in relations.relation_triples("unused")


def test_relations_for_corpus_deduplicates_results(monkeypatch):
    """Repeated relations across comments should only appear once."""

    monkeypatch.setattr(
        relations,
        "_tagged",
        lambda text: [("Rockstar", "NNP"), ("released", "VBD"), ("trailer", "NN")],
    )

    output = relations.relations_for_corpus(["one", "two"])

    assert output == ["released(Rockstar, trailer)"]


def test_noun_chunks_finds_multi_noun_phrase(monkeypatch):
    """Consecutive nouns should be grouped by the Lab 2 chunk grammar."""

    monkeypatch.setattr(
        relations,
        "_tagged",
        lambda text: [
            ("Grand", "NNP"),
            ("Theft", "NNP"),
            ("Auto", "NNP"),
            ("trailer", "NN"),
            ("impressed", "VBD"),
            ("fans", "NNS"),
        ],
    )

    chunks = relations.noun_chunks("unused")

    assert "Grand Theft Auto trailer" in chunks


def test_knowledge_graph_dot_contains_graph_edges(monkeypatch):
    """The DOT graph should contain nodes and labelled relation edges."""

    monkeypatch.setattr(
        relations,
        "_tagged",
        lambda text: [("Rockstar", "NNP"), ("released", "VBD"), ("trailer", "NN")],
    )

    dot = relations.knowledge_graph_dot(["unused"])

    assert dot.startswith("digraph KG")
    assert '"rockstar" -> "trailer" [label="released"];' in dot