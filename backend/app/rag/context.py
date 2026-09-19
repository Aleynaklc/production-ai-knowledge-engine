"""Token-budgeted context assembly with stable citation identifiers."""

from html import escape

from pydantic import BaseModel, ConfigDict, Field

from backend.app.ingestion.chunkers import TokenCodec
from backend.app.rag.evidence import factual_evidence
from backend.app.retrieval.models import RetrievalResult


class ContextSource(BaseModel):
    """One retrieved passage exposed to the language model."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    retrieval_score: float
    retrieval_rank: int = Field(ge=1)
    token_count: int = Field(ge=1)
    truncated: bool = False
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    document_version: int | None = None
    source_unit: int | None = None
    source_kind: str | None = None


class ContextBundle(BaseModel):
    """Rendered context plus its source map and budget accounting."""

    model_config = ConfigDict(frozen=True)

    query: str
    rendered: str
    sources: list[ContextSource]
    token_count: int = Field(ge=0)
    token_budget: int = Field(ge=1)
    skipped_source_count: int = Field(ge=0)
    format: str = "xml"


class ContextBuilder:
    """Select ranked passages without exceeding the language-model context budget."""

    def __init__(
        self,
        codec: TokenCodec,
        *,
        token_budget: int = 900,
        max_sources: int = 5,
        minimum_content_tokens: int = 8,
        preserve_document_order: bool = False,
        reading_format: bool = False,
    ) -> None:
        if token_budget < 32:
            raise ValueError("Context token budget must be at least 32")
        if max_sources < 1:
            raise ValueError("Context must allow at least one source")
        self.codec = codec
        self.token_budget = token_budget
        self.max_sources = max_sources
        self.minimum_content_tokens = minimum_content_tokens
        self.preserve_document_order = preserve_document_order
        self.reading_format = reading_format

    @staticmethod
    def _title(result: RetrievalResult) -> str:
        title = result.chunk.metadata.get("title")
        title = title if isinstance(title, str) else result.chunk.document_id
        heading = result.chunk.metadata.get("section_heading")
        return f"{title} — {heading}" if isinstance(heading, str) and heading else title

    def _render(self, citation_id: str, result: RetrievalResult, text: str) -> str:
        if self.reading_format:
            return f"File: {self._title(result)}\n{factual_evidence(text)}"
        return (
            f'<source id="{citation_id}" title="{escape(self._title(result), quote=True)}" '
            f'path="{escape(result.chunk.source, quote=True)}">\n'
            f"SOURCE [{citation_id}]\n"
            f"{escape(text)}\n"
            "</source>"
        )

    def _fit_text(
        self,
        citation_id: str,
        result: RetrievalResult,
        available_tokens: int,
    ) -> tuple[str, str, bool] | None:
        full_rendered = self._render(citation_id, result, result.chunk.text)
        if self.codec.count(full_rendered) <= available_tokens:
            return result.chunk.text, full_rendered, False

        offsets = self.codec.offsets(result.chunk.text)
        lower = 0
        upper = len(offsets)
        best: tuple[str, str] | None = None
        while lower <= upper:
            middle = (lower + upper) // 2
            if middle == 0:
                candidate_text = ""
            else:
                candidate_text = result.chunk.text[: offsets[middle - 1][1]].rstrip() + "…"
            rendered = self._render(citation_id, result, candidate_text)
            if self.codec.count(rendered) <= available_tokens:
                best = (candidate_text, rendered)
                lower = middle + 1
            else:
                upper = middle - 1
        if best is None or self.codec.count(best[0]) < self.minimum_content_tokens:
            return None
        return best[0], best[1], True

    def build(self, query: str, results: list[RetrievalResult]) -> ContextBundle:
        """Deduplicate, budget, and label retrieval results in rank order."""

        selected: list[ContextSource] = []
        rendered_blocks: list[str] = []
        seen_chunk_ids: set[str] = set()
        seen_texts: set[str] = set()
        skipped = 0

        for result in results:
            normalized_text = " ".join(result.chunk.text.casefold().split())
            if result.chunk.chunk_id in seen_chunk_ids or normalized_text in seen_texts:
                skipped += 1
                continue
            if len(selected) >= self.max_sources:
                skipped += 1
                continue

            separator = "\n\n" if rendered_blocks else ""
            used_tokens = self.codec.count(separator.join(rendered_blocks))
            separator_tokens = self.codec.count(separator)
            available = self.token_budget - used_tokens - separator_tokens
            citation_id = f"S{len(selected) + 1}"
            fitted = self._fit_text(citation_id, result, available)
            if fitted is None:
                skipped += 1
                continue
            text, rendered, truncated = fitted
            source_tokens = self.codec.count(rendered)
            selected.append(
                ContextSource(
                    citation_id=citation_id,
                    chunk_id=result.chunk.chunk_id,
                    document_id=result.chunk.document_id,
                    source=result.chunk.source,
                    title=self._title(result),
                    text=text,
                    retrieval_score=result.score,
                    retrieval_rank=result.rank,
                    token_count=source_tokens,
                    truncated=truncated,
                    supporting_chunk_ids=(
                        str(
                            result.chunk.metadata.get("supporting_chunk_ids", result.chunk.chunk_id)
                        ).split(",")
                    ),
                    document_version=(
                        int(value)
                        if isinstance(value := result.chunk.metadata.get("document_version"), int)
                        else None
                    ),
                    source_unit=(
                        int(value)
                        if isinstance(value := result.chunk.metadata.get("source_unit"), int)
                        else None
                    ),
                    source_kind=(
                        value
                        if isinstance(value := result.chunk.metadata.get("source_kind"), str)
                        else None
                    ),
                )
            )
            rendered_blocks.append(rendered)
            seen_chunk_ids.add(result.chunk.chunk_id)
            if not truncated:
                seen_chunk_ids.update(selected[-1].supporting_chunk_ids)
            seen_texts.add(normalized_text)

        if self.preserve_document_order:
            # Select by relevance, then read the selected passages in document order.
            # This reconnects headings/identity with adjacent education or experience text.
            document_order = list(dict.fromkeys(source.document_id for source in selected))
            result_by_id = {result.chunk.chunk_id: result for result in results}
            selected.sort(
                key=lambda source: (
                    document_order.index(source.document_id),
                    result_by_id[source.chunk_id].chunk.chunk_index,
                )
            )
            rendered_blocks = [
                self._render(source.citation_id, result_by_id[source.chunk_id], source.text)
                for source in selected
            ]
        rendered_context = "\n\n".join(rendered_blocks)
        return ContextBundle(
            query=query,
            rendered=rendered_context,
            sources=selected,
            token_count=self.codec.count(rendered_context),
            token_budget=self.token_budget,
            skipped_source_count=skipped,
            format="reading" if self.reading_format else "xml",
        )
