"""End-to-end retrieval, context construction, generation, and grounding gate."""

import re
from pathlib import Path
from time import perf_counter
from typing import Protocol

from backend.app.rag.attribution import _tokens, attribute_answer
from backend.app.rag.calculation import table_total
from backend.app.rag.citations import (
    CITATION_PATTERN,
    CitationValidation,
    number_tokens,
    validate_citations,
)
from backend.app.rag.context import ContextBuilder, ContextSource
from backend.app.rag.evidence import factual_evidence
from backend.app.rag.extraction import extract_requested_evidence, word_forms
from backend.app.rag.fallback import ExtractiveFallback
from backend.app.rag.generator import GroundedGenerator
from backend.app.rag.models import AnswerCitation, RAGAnswer, RAGStatus, RAGTiming
from backend.app.rag.prompt import (
    DOCUMENT_READING_SYSTEM_PROMPT,
    GROUNDED_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_RESPONSE,
    UNVERIFIED_ANSWER_RESPONSE,
    build_document_reading_prompt,
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
        supporting_chunk_ids=source.supporting_chunk_ids,
        retrieval_score=source.retrieval_score,
        document_version=source.document_version,
        source_unit=source.source_unit,
        source_kind=source.source_kind,
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
        attribute_sources: bool = False,
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
        self.attribute_sources = attribute_sources

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
            outcome_reason="no_evidence",
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
        retrieval_depth = self.default_top_k if top_k is None else top_k
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

        calculation_started = perf_counter()
        extraction = (
            extract_requested_evidence(normalized_question, context)
            if self.attribute_sources
            else None
        )
        calculation = table_total(normalized_question, context) if self.attribute_sources else None
        if calculation is not None or extraction is not None:
            sources = {source.citation_id: source for source in context.sources}
            citation_ids = (
                [calculation.citation_id]
                if calculation
                else list(dict.fromkeys(item.citation_id for item in extraction.excerpts))
                if extraction
                else []
            )
            validation = _abstention_validation().model_copy(
                update={"abstained": False, "cited_source_ids": citation_ids}
            )
            return RAGAnswer(
                question=normalized_question,
                status="answered",
                answer=calculation.answer
                if calculation
                else extraction.answer
                if extraction
                else "",
                raw_answer=None,
                fallback_used=False,
                calculation=calculation,
                extraction=extraction,
                citations=[_citation(sources[citation_id]) for citation_id in citation_ids],
                retrieved_sources=[_citation(source) for source in context.sources],
                validation=validation,
                context_source_count=len(context.sources),
                context_token_count=context.token_count,
                input_tokens=0,
                output_tokens=0,
                timings=RAGTiming(
                    retrieval_ms=retrieval_ms,
                    context_ms=context_ms,
                    generation_ms=0,
                    grounding_ms=(perf_counter() - calculation_started) * 1000,
                    total_ms=(perf_counter() - total_started) * 1000,
                ),
            )

        prompt = (
            build_document_reading_prompt(normalized_question, context)
            if self.attribute_sources
            else build_grounded_prompt(normalized_question, context)
        )
        system_prompt = (
            DOCUMENT_READING_SYSTEM_PROMPT if self.attribute_sources else GROUNDED_SYSTEM_PROMPT
        )
        generation = self.generator.generate(prompt, system_prompt)
        grounding_started = perf_counter()
        raw_answer = generation.text.strip()
        generated_answer = (
            attribute_answer(raw_answer, context) if self.attribute_sources else raw_answer
        )
        validation = validate_citations(generated_answer, context)
        # Explicit temporal qualifiers must be present in the cited evidence.
        # A current allowance is not evidence for an unmentioned future policy.
        requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", normalized_question))
        cited_text = "\n".join(
            source.text
            for source in context.sources
            if source.citation_id in validation.cited_source_ids
        )
        plain_answer = CITATION_PATTERN.sub("", raw_answer).strip()
        claim_tokens = _tokens(plain_answer)
        quantity_words = {
            "day",
            "year",
            "month",
            "week",
            "hour",
            "minute",
            "second",
            "usd",
            "eur",
            "gbp",
            "tl",
            "percent",
        }
        quantity_words |= {word + "s" for word in quantity_words}
        quantity_only = (
            bool(claim_tokens)
            and claim_tokens.issubset(number_tokens(plain_answer) | quantity_words)
            and len(number_tokens(plain_answer)) == 1
        )
        question_terms = _tokens(normalized_question) - {
            "what",
            "which",
            "how",
            "much",
            "many",
            "long",
            "do",
            "doe",
            "did",
            "under",
            "please",
            "document",
            "setting",
            "stock",
            "should",
            "be",
            "often",
            "does",
        }
        # A document-name qualifier identifies the source, not a property of the number.
        # Splitting sections must not remove a filename already shown to the model.
        document_terms = set().union(
            *(
                _tokens(Path(source.source).stem)
                for source in context.sources
                if source.citation_id in validation.cited_source_ids
            )
        )
        question_terms -= {term for term in document_terms if term.isalpha()}
        question_terms = {"cost" if token == "fee" else token for token in question_terms}
        evidence_terms = {
            "cost" if token == "fee" else token for token in _tokens(factual_evidence(cited_text))
        }
        evidence_forms = set().union(*(word_forms(word) for word in evidence_terms))
        if quantity_only and any(not word_forms(word) & evidence_forms for word in question_terms):
            validation = validation.model_copy(
                update={
                    "valid": False,
                    "issues": [*validation.issues, "quantity_question_not_supported"],
                }
            )
        if not validation.abstained and requested_years - set(
            re.findall(r"\b(?:19|20)\d{2}\b", cited_text)
        ):
            validation = validation.model_copy(
                update={
                    "valid": False,
                    "issues": [*validation.issues, "question_date_not_supported"],
                }
            )
        if self.attribute_sources and generation.output_tokens >= generation.options.max_new_tokens:
            validation = validation.model_copy(
                update={
                    "valid": False,
                    "abstained": False,
                    "issues": [*validation.issues, "generation_reached_token_limit"],
                }
            )
        source_by_id = {source.citation_id: source for source in context.sources}
        fallback_used = False
        if (
            (validation.abstained or not validation.valid)
            and self.extractive_fallback is not None
            and not requested_years
            and "quantity_question_not_supported" not in validation.issues
        ):
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
            answer = UNVERIFIED_ANSWER_RESPONSE
            citations = []
        else:
            status = "answered"
            answer = generated_answer
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
            retrieved_sources=[_citation(source) for source in context.sources],
            outcome_reason=(
                None
                if status == "answered"
                else "generation_limit"
                if "generation_reached_token_limit" in validation.issues
                else "not_in_sources"
                if status == "abstained"
                else "verification_failed"
            ),
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
            generation_provider=generation.provider,
            generation_model=generation.model,
        )
