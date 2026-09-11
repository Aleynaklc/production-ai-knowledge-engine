"""End-to-end retrieval, context construction, generation, and grounding gate."""

from time import perf_counter
from typing import Protocol

from backend.app.rag.citations import CitationValidation, validate_citations
from backend.app.rag.context import ContextBuilder, ContextSource
from backend.app.rag.fallback import ExtractiveFallback
from backend.app.rag.generator import GroundedGenerator
from backend.app.rag.models import AnswerCitation, RAGAnswer, RAGStatus, RAGTiming
from backend.app.rag.prompt import (
    GROUNDED_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_RESPONSE,
    build_grounded_prompt,
)
from backend.app.retrieval.models import Retriever


class AnswerService(Protocol):
    """Boundary consumed by the HTTP API and evaluation harness."""

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Return one safe grounded answer."""


def _abstention_validation() -> CitationValidation:
    return CitationValidation(
        valid=True,
        abstained=True,
        cited_source_ids=[],
        unknown_source_ids=[],
        uncited_claims=[],
        unsupported_claims=[],
        issues=[],
    )


def _citation(source: ContextSource) -> AnswerCitation:
    snippet = source.text if len(source.text) <= 320 else source.text[:319].rstrip() + "…"
    return AnswerCitation(
        citation_id=source.citation_id,
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        source=source.source,
        title=source.title,
        snippet=snippet,
        retrieval_score=source.retrieval_score,
    )


class RAGService:
    """Run grounded RAG and suppress answers that fail citation validation."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: GroundedGenerator,
        *,
        default_top_k: int = 5,
        strict_grounding: bool = True,
        minimum_retrieval_score: float | None = None,
        extractive_fallback: ExtractiveFallback | None = None,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be positive")
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator
        self.default_top_k = default_top_k
        self.strict_grounding = strict_grounding
        self.minimum_retrieval_score = minimum_retrieval_score
        self.extractive_fallback = extractive_fallback

    def _empty_answer(
        self,
        question: str,
        retrieval_ms: float,
        total_ms: float,
        context_ms: float = 0.0,
    ) -> RAGAnswer:
        return RAGAnswer(
            question=question,
            status="abstained",
            answer=INSUFFICIENT_CONTEXT_RESPONSE,
            raw_answer=None,
            fallback_used=False,
            citations=[],
            validation=_abstention_validation(),
            context_source_count=0,
            context_token_count=0,
            timings=RAGTiming(
                retrieval_ms=round(retrieval_ms, 6),
                context_ms=round(context_ms, 6),
                generation_ms=0.0,
                grounding_ms=0.0,
                total_ms=round(total_ms, 6),
            ),
            input_tokens=0,
            output_tokens=0,
        )

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Answer from retrieved evidence or safely abstain/reject."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty")
        retrieval_depth = top_k or self.default_top_k
        if retrieval_depth < 1:
            raise ValueError("top_k must be positive")

        total_started = perf_counter()
        retrieval_started = perf_counter()
        results = self.retriever.retrieve(normalized_question, top_k=retrieval_depth)
        retrieval_ms = (perf_counter() - retrieval_started) * 1_000
        if self.minimum_retrieval_score is not None and (
            not results or results[0].score < self.minimum_retrieval_score
        ):
            total_ms = (perf_counter() - total_started) * 1_000
            return self._empty_answer(normalized_question, retrieval_ms, total_ms)
        context_started = perf_counter()
        context = self.context_builder.build(normalized_question, results)
        context_ms = (perf_counter() - context_started) * 1_000
        if not context.sources:
            total_ms = (perf_counter() - total_started) * 1_000
            return self._empty_answer(normalized_question, retrieval_ms, total_ms, context_ms)

        prompt = build_grounded_prompt(normalized_question, context)
        generation = self.generator.generate(prompt, GROUNDED_SYSTEM_PROMPT)
        grounding_started = perf_counter()
        raw_answer = generation.text.strip()
        validation = validate_citations(raw_answer, context)
        source_by_id = {source.citation_id: source for source in context.sources}
        generated_answer = raw_answer
        fallback_used = False
        if (validation.abstained or not validation.valid) and self.extractive_fallback is not None:
            fallback_answer = self.extractive_fallback.answer(normalized_question, context)
            if fallback_answer is not None:
                fallback_validation = validate_citations(fallback_answer, context)
                if fallback_validation.valid:
                    generated_answer = fallback_answer
                    validation = fallback_validation
                    fallback_used = True

        status: RAGStatus
        citations: list[AnswerCitation]
        if fallback_used:
            status = "answered"
            answer = generated_answer
            citations = [
                _citation(source_by_id[source_id]) for source_id in validation.cited_source_ids
            ]
        elif validation.abstained:
            status = "abstained"
            answer = INSUFFICIENT_CONTEXT_RESPONSE
            citations = []
        elif validation.valid:
            status = "answered"
            answer = generated_answer
            citations = [
                _citation(source_by_id[source_id]) for source_id in validation.cited_source_ids
            ]
        elif self.strict_grounding:
            status = "rejected"
            answer = INSUFFICIENT_CONTEXT_RESPONSE
            citations = []
        else:
            status = "answered"
            answer = raw_answer
            citations = [
                _citation(source_by_id[source_id])
                for source_id in validation.cited_source_ids
                if source_id in source_by_id
            ]

        grounding_ms = (perf_counter() - grounding_started) * 1_000
        total_ms = (perf_counter() - total_started) * 1_000
        return RAGAnswer(
            question=normalized_question,
            status=status,
            answer=answer,
            raw_answer=raw_answer,
            fallback_used=fallback_used,
            citations=citations,
            validation=validation,
            context_source_count=len(context.sources),
            context_token_count=context.token_count,
            timings=RAGTiming(
                retrieval_ms=round(retrieval_ms, 6),
                context_ms=round(context_ms, 6),
                generation_ms=round(generation.generation_seconds * 1_000, 6),
                grounding_ms=round(grounding_ms, 6),
                total_ms=round(total_ms, 6),
            ),
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
        )
