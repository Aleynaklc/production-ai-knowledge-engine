"""Attach evidence to generated claims, without asking the model to invent source labels.

This is a conservative lexical check, not a semantic entailment model. The regular
citation validator still runs afterwards. Unmatched claims remain uncited and fail closed.
"""

import unicodedata
from itertools import combinations

from backend.app.rag.citations import (
    CITATION_PATTERN,
    TOKEN_PATTERN,
    _claims,
    _content_tokens,
    number_tokens,
)
from backend.app.rag.context import ContextBundle
from backend.app.rag.evidence import factual_evidence
from backend.app.rag.prompt import INSUFFICIENT_CONTEXT_RESPONSE
from backend.app.retrieval.sparse import lexical_form


def _tokens(text: str) -> set[str]:
    # Match typographic accents/case without changing the displayed document or answer.
    normalized = unicodedata.normalize("NFKD", text.casefold().replace("ı", "i"))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return {lexical_form(token) for token in _content_tokens(normalized)} - {
        "she",
        "he",
        "her",
        "his",
        "they",
        "their",
        "there",
        "while",
        "whereas",
        "ve",
    }


def attribute_answer(answer: str, context: ContextBundle) -> str:
    """Use the smallest supporting set of up to three sources for each complete claim."""
    if answer.strip() == INSUFFICIENT_CONTEXT_RESPONSE:
        return answer
    sources = context.sources
    tokens = {source.citation_id: _tokens(factual_evidence(source.text)) for source in sources}
    result = []
    for claim in _claims(answer):
        supplied = {"S" + value for value in CITATION_PATTERN.findall(claim)}
        if supplied - set(tokens):
            result.append(claim)  # Keep unknown labels visible to the citation validator.
            continue
        claim = CITATION_PATTERN.sub("", claim).strip()
        claim_tokens = _tokens(claim)
        numbers = number_tokens(claim)
        words = TOKEN_PATTERN.findall(claim)
        named_tokens = set().union(*(_tokens(word) for word in words[1:] if word[0].isupper()))
        # Very short claims (e.g. a person's name) must match every content token.
        required = 1.0 if len(claim_tokens) <= 3 else 0.75
        labels: list[str] = []
        for size in range(1, min(3, len(sources)) + 1):
            best_coverage = 0.0
            for group in combinations(sources, size):
                if supplied and supplied != {source.citation_id for source in group}:
                    continue
                evidence = " ".join(factual_evidence(source.text) for source in group)
                if not numbers.issubset(number_tokens(evidence)):
                    continue
                evidence_tokens = set().union(*(tokens[source.citation_id] for source in group))
                if not named_tokens.issubset(evidence_tokens):
                    continue
                # Do not attach a positive source to a newly introduced negative claim.
                if (claim_tokens & {"not", "never", "no"}) - evidence_tokens:
                    continue
                coverage = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
                if coverage >= required and coverage > best_coverage:
                    labels = [source.citation_id for source in group]
                    best_coverage = coverage
            if labels:
                break
        prefix = "".join(f"[{label}]" for label in labels)
        result.append(f"{prefix} {claim}".strip())
    return "\n".join(result)
