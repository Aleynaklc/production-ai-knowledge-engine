"""Verified source excerpts for explicit example and ordered-procedure requests."""

import re

from pydantic import BaseModel, ConfigDict

from backend.app.rag.context import ContextBundle
from backend.app.rag.evidence import factual_evidence
from backend.app.retrieval.sparse import tokenize_for_bm25
from backend.app.retrieval.workspace import section_headings


class SourceExcerpt(BaseModel):
    model_config = ConfigDict(frozen=True)
    citation_id: str
    text: str


class EvidenceExtraction(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: str
    excerpts: list[SourceExcerpt]

    @property
    def answer(self) -> str:
        return "\n\n".join(f"[{item.citation_id}] {item.text}" for item in self.excerpts)


def word_forms(word: str) -> set[str]:
    """Small grammatical equivalences used only for evidence matching."""
    forms = {word}
    for suffix in ("ed", "ing", "ment"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            stem = word[: -len(suffix)]
            forms.update((stem, stem + "e"))
    return forms


def extract_requested_evidence(question: str, context: ContextBundle) -> EvidenceExtraction | None:
    words = set(tokenize_for_bm25(question))
    examples = bool(words & {"example", "examples", "ornek", "ornekler", "ornegi", "ornekleri"})
    # Explanations/comparisons and requested modifications still need generation.
    if words & {
        "explain",
        "why",
        "compare",
        "modify",
        "change",
        "optimize",
        "acikla",
        "neden",
        "without",
        "only",
        "except",
        "not",
        "sadece",
        "haric",
    }:
        return None
    if examples and words & {"give", "show", "list", "provide", "ver", "goster", "listele"}:
        excerpts: list[SourceExcerpt] = []
        seen: set[str] = set()
        limit = 1 if words & {"one", "bir", "1"} else 2 if words & {"two", "iki", "2"} else 3
        for source in context.sources:
            for match in re.finditer(r"(?m)^```[^\n]*\n[\s\S]*?^```\s*$", source.text):
                code = match[0].strip()
                if code in seen:
                    continue
                seen.add(code)
                # Include the nearest source heading when it is in this actual excerpt.
                before = source.text[: match.start()]
                headings = section_headings(before)
                heading = "## " + headings[-1] if headings else ""
                title_words = set(
                    tokenize_for_bm25(
                        re.sub(r"\d+", "", source.source + " " + source.title)
                        + " "
                        + (headings[0] if headings else "")
                    )
                )
                topic_words = (
                    words
                    - title_words
                    - {
                        "give",
                        "show",
                        "list",
                        "provide",
                        "ver",
                        "goster",
                        "listele",
                        "example",
                        "examples",
                        "ornek",
                        "ornekler",
                        "ornegi",
                        "ornekleri",
                        "from",
                        "in",
                        "of",
                        "the",
                        "a",
                        "an",
                        "some",
                        "me",
                        "please",
                        "question",
                        "questions",
                        "document",
                        "two",
                        "three",
                        "one",
                        "iki",
                        "uc",
                        "bir",
                        "soru",
                        "belgesinden",
                        "belgede",
                        "1",
                        "2",
                        "3",
                    }
                )
                code_words = set(tokenize_for_bm25(heading + " " + code))
                if topic_words and not topic_words.issubset(code_words):
                    continue
                text = heading + "\n" + code if heading else code
                excerpts.append(SourceExcerpt(citation_id=source.citation_id, text=text))
                if len(excerpts) == limit:
                    return EvidenceExtraction(kind="code_examples", excerpts=excerpts)
        return EvidenceExtraction(kind="code_examples", excerpts=excerpts) if excerpts else None
    if not (words & {"how", "steps", "step", "procedure", "nasil", "adim", "adimlar"}):
        return None
    question_terms = words - {
        "how",
        "what",
        "which",
        "should",
        "be",
        "the",
        "a",
        "an",
        "to",
        "are",
        "is",
        "steps",
        "step",
        "procedure",
        "please",
        "nasil",
        "adim",
        "adimlar",
    }
    candidates: dict[str, EvidenceExtraction] = {}
    for source in context.sources:
        text = factual_evidence(source.text)
        for match in re.finditer(r"(?m)(?:^\d+[.)]\s+[^\n]+\n?){2,}", text):
            before = text[: match.start()]
            procedure_headings = section_headings(before)
            heading = procedure_headings[-1] if procedure_headings else source.title
            evidence_words = set(tokenize_for_bm25(heading + " " + source.title + " " + match[0]))
            forms = set().union(*(word_forms(word) for word in evidence_words))
            if not question_terms or any(not word_forms(word) & forms for word in question_terms):
                continue
            excerpts = [
                SourceExcerpt(citation_id=source.citation_id, text=line.strip())
                for line in match[0].splitlines()
                if line.strip()
            ]
            candidates.setdefault(match[0], EvidenceExtraction(kind="procedure", excerpts=excerpts))
    return next(iter(candidates.values())) if len(candidates) == 1 else None
