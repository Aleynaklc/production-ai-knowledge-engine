"""Document-scoped, section-diverse retrieval over an already authorized workspace.

Request categories change evidence selection only. No questions or answer facts are stored
here; every returned passage is an original active chunk from the supplied workspace.
"""

import re
from dataclasses import replace
from pathlib import Path

from backend.app.ingestion.models import DocumentChunk
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.models import RetrievalResult, Retriever
from backend.app.retrieval.reranker import PairScorer
from backend.app.retrieval.sparse import BM25Retriever, tokenize_for_bm25

HEADINGS = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
EXAMPLES = {"example", "examples", "ornek", "ornekler", "ornegi", "ornekleri"}
OVERVIEW = {"summary", "summarize", "overview", "ozet", "ozetle", "summarise"}
COMPARISON = {"compare", "comparison", "difference", "differences", "karsilastir", "farki"}
INDEX_WORDS = {"contents", "introduction", "index", "dizin", "kapsam", "amac", "giris"}


def section_headings(text: str) -> list[str]:
    """Python comments inside fenced examples are not Markdown section headings."""
    headings: list[str] = []
    fence = ""
    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            fence = "" if fence == marker[1] else marker[1] if not fence else fence
            continue
        if not fence and (heading := HEADINGS.match(line)):
            headings.append(heading[1].strip())
    return headings


def _name_tokens(source: str) -> set[str]:
    return set(tokenize_for_bm25(re.sub(r"\d+", " ", Path(source).stem))) - {
        "the",
        "and",
        "for",
        "with",
        "how",
        "what",
        "are",
        "can",
        "you",
        "bir",
        "icin",
        "in",
        "to",
        "of",
        "ve",
        "bu",
    }


class WorkspaceRetriever:
    name = "workspace_hybrid_reranked"

    def __init__(
        self,
        dense: Retriever,
        chunks: list[DocumentChunk],
        scorer: PairScorer,
        candidate_k: int = 20,
        *,
        score_gap: float | None = None,
    ) -> None:
        if candidate_k < 1 or not chunks:
            raise ValueError("Workspace retrieval needs chunks and a positive candidate limit")
        if score_gap is not None and score_gap <= 0:
            raise ValueError("score_gap must be positive")
        self.score_gap = score_gap
        self.dense, self.scorer, self.candidate_k = dense, scorer, candidate_k
        self.chunks: dict[str, DocumentChunk] = {}
        self.documents: dict[str, list[DocumentChunk]] = {}
        for chunk in sorted(chunks, key=lambda item: (item.document_id, item.chunk_index)):
            siblings = self.documents.setdefault(chunk.document_id, [])
            headings = section_headings(chunk.text)
            previous_headings = section_headings(siblings[-1].text) if siblings else []
            previous = (
                previous_headings[-1]
                if previous_headings
                else siblings[-1].metadata.get("section_heading", "")
                if siblings
                else ""
            )
            heading = headings[0].strip() if headings else previous
            enriched = chunk.model_copy(
                update={
                    "metadata": {
                        **chunk.metadata,
                        "section_heading": heading,
                    }
                }
            )
            siblings.append(enriched)
            self.chunks[chunk.chunk_id] = enriched
        self.lexical = BM25Retriever(list(self.chunks.values()), include_metadata=True)

    def _with_neighbors(self, item: RetrievalResult) -> RetrievalResult:
        """Reconnect a split section without crossing document/version/page boundaries."""
        anchor = item.chunk
        siblings = self.documents[anchor.document_id]
        index = next(i for i, chunk in enumerate(siblings) if chunk.chunk_id == anchor.chunk_id)
        neighbors = [
            chunk
            for chunk in siblings[max(0, index - 1) : index + 2]
            if chunk.metadata.get("source_unit") == anchor.metadata.get("source_unit")
            and chunk.metadata.get("document_version") == anchor.metadata.get("document_version")
            and self._section(chunk) == self._section(anchor)
        ]
        if len(neighbors) <= 1:
            return item
        text = neighbors[0].text
        for chunk in neighbors[1:]:
            # Remove exact copied overlap, otherwise keep an explicit paragraph boundary.
            overlap = next(
                (
                    size
                    for size in range(min(len(text), len(chunk.text)), 7, -1)
                    if text.endswith(chunk.text[:size])
                ),
                0,
            )
            text += chunk.text[overlap:] if overlap else "\n\n" + chunk.text
        return replace(
            item,
            chunk=anchor.model_copy(
                update={
                    "text": text,
                    "metadata": {
                        **anchor.metadata,
                        "supporting_chunk_ids": ",".join(chunk.chunk_id for chunk in neighbors),
                    },
                }
            ),
        )

    def _scope(self, query: str) -> set[str]:
        words = set(tokenize_for_bm25(query))
        # Resolve document names from this corpus, not a product-specific vocabulary.
        matches = {
            document_id: len(words & _name_tokens(chunks[0].source))
            for document_id, chunks in self.documents.items()
        }
        return {key for key, count in matches.items() if count > 0}

    @staticmethod
    def _section(chunk: DocumentChunk) -> tuple[str, str]:
        heading = str(chunk.metadata.get("section_heading", ""))
        return chunk.document_id, heading or str(
            chunk.metadata.get("source_unit", chunk.chunk_index)
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        words = set(tokenize_for_bm25(query))
        examples = bool(words & EXAMPLES)
        diverse = examples or bool(words & (OVERVIEW | COMPARISON))
        scope = self._scope(query)
        depth = max(self.candidate_k, top_k)
        dense = self.dense.retrieve(query, depth)
        lexical = self.lexical.retrieve(query, depth)
        # A filename mention is a recall hint, not authorization to discard other
        # documents ("Python skills" can concern a CV beside a Python manual).
        fused = reciprocal_rank_fusion([dense, lexical], top_k=depth * 2)
        candidates = {
            item.chunk.chunk_id: replace(item, chunk=self.chunks[item.chunk.chunk_id])
            for item in fused
            if item.chunk.chunk_id in self.chunks
        }
        if len(self.chunks) <= depth:
            for chunk in self.chunks.values():
                candidates.setdefault(
                    chunk.chunk_id, RetrievalResult(chunk, 0, len(candidates) + 1, self.name)
                )
        for document_id in scope:
            for chunk in self.documents[document_id][:depth]:
                candidates.setdefault(
                    chunk.chunk_id, RetrievalResult(chunk, 0, len(candidates) + 1, self.name)
                )
        if diverse:
            sections: set[tuple[str, str]] = set()
            for chunk in self.chunks.values():
                heading_words = set(
                    tokenize_for_bm25(str(chunk.metadata.get("section_heading", "")))
                )
                if examples and (
                    heading_words & INDEX_WORDS
                    or not ("```" in chunk.text or set(tokenize_for_bm25(chunk.text)) & EXAMPLES)
                ):
                    continue
                section = self._section(chunk)
                if section in sections:
                    continue
                sections.add(section)
                candidates.setdefault(
                    chunk.chunk_id, RetrievalResult(chunk, 0, len(candidates) + 1, self.name)
                )
                if len(candidates) >= depth * 3:
                    break
        pool = list(candidates.values())
        if not pool:
            return []
        scores = self.scorer.score(
            query,
            [
                f"Document: {item.chunk.source}\nSection: {item.chunk.metadata.get('section_heading', '')}\n{item.chunk.text}"
                for item in pool
            ],
        )
        if len(scores) != len(pool):
            raise ValueError("Unexpected workspace reranker score count")
        ranked = [
            replace(
                item, score=score, component_scores={**item.component_scores, "reranker": score}
            )
            for item, score in zip(pool, scores, strict=True)
        ]

        def priority(item: RetrievalResult) -> tuple[float, int]:
            heading = set(tokenize_for_bm25(str(item.chunk.metadata.get("section_heading", ""))))
            concrete = "```" in item.chunk.text or bool(
                set(tokenize_for_bm25(item.chunk.text)) & EXAMPLES
            )
            # Structural hints break close rankings, never override relevance outright.
            bonus = 0.5 if examples and concrete and not heading & INDEX_WORDS else 0.0
            return (item.score + bonus, int(item.chunk.document_id in scope))

        if self.score_gap is not None:
            best_score = max(item.score for item in ranked)
            ranked = [item for item in ranked if item.score >= best_score - self.score_gap]
        ranked.sort(key=priority, reverse=True)
        selected: list[RetrievalResult] = []
        sections = set()
        for item in ranked:
            section = self._section(item.chunk)
            if diverse and section in sections:
                continue
            selected.append(item)
            sections.add(section)
            if len(selected) >= top_k:
                break
        # Fill remaining slots when fewer distinct sections are available.
        selected_ids = {item.chunk.chunk_id for item in selected}
        for item in ranked:
            if len(selected) >= top_k:
                break
            if item.chunk.chunk_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.chunk.chunk_id)
        return [
            replace(self._with_neighbors(item), rank=rank, retriever=self.name)
            for rank, item in enumerate(selected, 1)
        ]
