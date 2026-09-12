"""Parse and validate source citations before an answer reaches the user."""

import re

from pydantic import BaseModel, ConfigDict

from backend.app.rag.context import ContextBundle
from backend.app.rag.prompt import INSUFFICIENT_CONTEXT_RESPONSE

CITATION_PATTERN = re.compile(r"\[S([1-9][0-9]*)\]")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")
TOKEN_PATTERN = re.compile(r"[^\W_]+")
NUMBER_PATTERN = re.compile(r"\b\d+(?::\d+)?\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


class CitationValidation(BaseModel):
    """Machine-readable grounding decision and its failure reasons."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    abstained: bool
    cited_source_ids: list[str]
    unknown_source_ids: list[str]
    uncited_claims: list[str]
    unsupported_claims: list[str]
    issues: list[str]


def extract_citation_ids(answer: str) -> list[str]:
    """Return unique citation identifiers in their first-appearance order."""

    return list(dict.fromkeys(f"S{match}" for match in CITATION_PATTERN.findall(answer)))


def _claims(answer: str) -> list[str]:
    parts = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(answer.strip())
        if sentence.strip() and any(character.isalnum() for character in sentence)
    ]
    claims: list[str] = []
    for part in parts:
        if CITATION_PATTERN.fullmatch(part) and claims:
            claims[-1] = f"{claims[-1]} {part}"
        else:
            claims.append(part)
    return claims


def _content_tokens(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.casefold()) if token not in STOPWORDS}


def _claim_is_supported(claim: str, context: ContextBundle) -> bool:
    citation_ids = extract_citation_ids(claim)
    source_by_id = {source.citation_id: source for source in context.sources}
    evidence = " ".join(
        source_by_id[citation_id].text
        for citation_id in citation_ids
        if citation_id in source_by_id
    )
    plain_claim = CITATION_PATTERN.sub("", claim).strip()
    claim_tokens = _content_tokens(plain_claim)
    if not claim_tokens or not evidence:
        return False
    claim_numbers = set(NUMBER_PATTERN.findall(plain_claim))
    evidence_numbers = set(NUMBER_PATTERN.findall(evidence))
    if not claim_numbers.issubset(evidence_numbers):
        return False
    evidence_tokens = _content_tokens(evidence)
    lexical_coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    return lexical_coverage >= 0.35


def validate_citations(answer: str, context: ContextBundle) -> CitationValidation:
    """Reject unknown citations and factual sentences without source attribution."""

    normalized_answer = " ".join(answer.split())
    if normalized_answer == INSUFFICIENT_CONTEXT_RESPONSE:
        return CitationValidation(
            valid=True,
            abstained=True,
            cited_source_ids=[],
            unknown_source_ids=[],
            uncited_claims=[],
            unsupported_claims=[],
            issues=[],
        )

    cited = extract_citation_ids(answer)
    allowed = {source.citation_id for source in context.sources}
    unknown = [citation_id for citation_id in cited if citation_id not in allowed]
    claims = _claims(answer)
    substantive_claims = [
        claim for claim in claims if _content_tokens(CITATION_PATTERN.sub("", claim))
    ]
    uncited = [claim for claim in substantive_claims if not CITATION_PATTERN.search(claim)]
    unsupported = [
        claim
        for claim in substantive_claims
        if CITATION_PATTERN.search(claim) and not _claim_is_supported(claim, context)
    ]
    issues: list[str] = []
    if not cited:
        issues.append("answer_has_no_citations")
    if not substantive_claims:
        issues.append("answer_has_no_substantive_claims")
    if unknown:
        issues.append("answer_has_unknown_citations")
    if uncited:
        issues.append("answer_has_uncited_claims")
    if unsupported:
        issues.append("answer_has_unsupported_claims")
    return CitationValidation(
        valid=not issues,
        abstained=False,
        cited_source_ids=cited,
        unknown_source_ids=unknown,
        uncited_claims=uncited,
        unsupported_claims=unsupported,
        issues=issues,
    )
