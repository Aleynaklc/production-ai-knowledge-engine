"""A bounded table operation: explicit quantity times unit price, with provenance."""

import re
from decimal import Decimal, localcontext

from pydantic import BaseModel, ConfigDict

from backend.app.rag.context import ContextBundle
from backend.app.rag.evidence import factual_evidence
from backend.app.retrieval.sparse import tokenize_for_bm25


class CalculationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: str = "quantity_times_unit_price"
    item: str
    quantity: str
    unit_price: str
    result: str
    citation_id: str
    source_row: str

    @property
    def answer(self) -> str:
        return f"[{self.citation_id}] {self.item}: {self.quantity} × {self.unit_price} = {self.result} (quantity × unit price)."


def table_total(question: str, context: ContextBundle) -> CalculationEvidence | None:
    """Only calculate a uniquely named row and explicitly requested total.

    No arbitrary code, currency conversion, tax, discounts, inferred columns, or
    cross-table joins. Unsupported/ambiguous requests continue through normal RAG.
    """
    words = set(tokenize_for_bm25(question))
    if "total" not in words or not words & {"value", "cost", "price"}:
        return None
    grammar = {
        "what",
        "is",
        "the",
        "total",
        "value",
        "cost",
        "price",
        "of",
        "all",
        "in",
        "stock",
        "for",
        "calculate",
        "compute",
        "please",
    }
    candidates: dict[tuple[str, str, str, str], CalculationEvidence] = {}
    for source in context.sources:
        lines = factual_evidence(source.text).splitlines()
        header: list[str] = []
        for index, line in enumerate(lines):
            if not line.strip().startswith("|"):
                header = []
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if index + 1 < len(lines) and re.fullmatch(r"[\s|:\-]+", lines[index + 1]):
                header = [" ".join(tokenize_for_bm25(cell)) for cell in cells]
                continue
            if not header or len(cells) != len(header) or re.fullmatch(r"[\s|:\-]+", line):
                continue
            qty_columns = [i for i, name in enumerate(header) if name in {"quantity", "qty"}]
            price_columns = [
                i for i, name in enumerate(header) if name in {"unit price", "unit cost"}
            ]
            item_columns = [i for i, name in enumerate(header) if name in {"item", "product"}]
            if not (len(qty_columns) == len(price_columns) == len(item_columns) == 1):
                continue
            item = cells[item_columns[0]]
            item_words = set(tokenize_for_bm25(item))
            item_forms = item_words | {word + "s" for word in item_words}
            if not item_words or not all(
                word in words or word + "s" in words for word in item_words
            ):
                continue
            if words - grammar - item_forms:
                continue
            qty, price = cells[qty_columns[0]], cells[price_columns[0]]
            if not all(re.fullmatch(r"\d{1,15}(?:\.\d{1,8})?", value) for value in (qty, price)):
                continue
            with localcontext() as decimal_context:
                decimal_context.prec = 64
                value = format(Decimal(qty) * Decimal(price), "f")
            value = value.rstrip("0").rstrip(".") if "." in value else value
            key = (source.document_id, item, qty, price)
            candidates.setdefault(
                key,
                CalculationEvidence(
                    item=item,
                    quantity=qty,
                    unit_price=price,
                    result=value,
                    citation_id=source.citation_id,
                    source_row=line.strip(),
                ),
            )
    return next(iter(candidates.values())) if len(candidates) == 1 else None
