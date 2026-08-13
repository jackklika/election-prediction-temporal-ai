"""Tier 1 of the poll pipeline: a model rewrites what strict parsing refused.

The contract that keeps this safe is narrow on both ends:

- **In:** one cell the strict parser refused — a date like "Jul. 24–8, 2026",
  a sample like "~600 LV". Never a percentage: numbers travel from the DOM to
  the database untouched by any model, and a refused percentage stays refused.
- **Out:** a *canonical rewrite of the same text*, which the strict parser then
  re-parses. The model cannot produce a value — only a string the deterministic
  tier must still accept. A rewrite that does not strict-parse leaves the
  refusal standing. Null means "the text does not actually state this", which
  is the answer that makes hallucination structurally unattractive.

Anything the model touched is recorded: the revision's origin becomes MODEL
rather than IMPORT, so review can triage an interpreted date differently from
a parsed one. That distinction is why `RecordOrigin` has both members.

Cost profile: this runs only on refusals. On the five-race corpus the strict
tier covers everything, so a normal import makes zero API calls; this exists
for the page that drifts.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

from predictelection.clients.anthropic import AnthropicConfig


logger = logging.getLogger(__name__)


CANONICAL_FORMS: dict[str, str] = {
    "fieldwork_dates": (
        'a date or date range exactly like "July 24–28, 2026", '
        '"June 29 – July 29, 2026" or "May 5, 2026" (en dash between days)'
    ),
    "sample": 'a count and population code exactly like "600 (LV)" or "1,024 (RV)"',
    "margin": 'a margin exactly like "± 4.0%"',
}
"""What each cell kind must be rewritten into. These are the forms the strict
parsers accept; the model is aimed at them rather than asked to interpret."""


class CellNormalizer(Protocol):
    """The seam the importer calls on a refusal. No implementation required:

    an importer constructed without one runs fully deterministic, and refused
    rows fail visibly — the CLI's default, and what CI fixtures exercise.
    """

    def rewrite(self, kind: str, text: str) -> str | None: ...


class Rewrite(BaseModel):
    canonical: str | None = Field(
        description=(
            "The same information in the canonical form, or null if the text "
            "does not fully state it. Never fill in a missing month, year or "
            "count — null is the correct answer whenever anything is missing."
        )
    )


_INSTRUCTIONS = """\
You normalize one messy table cell from a Wikipedia opinion-polling table into
a canonical form. You are a formatter, not a source: every piece of the output
must be present in the input text. If the input does not fully state the value
— a date without a year, a sample described only as "adults", an approximate
count — return null rather than completing it. Strip footnote markers and
citations. Do not convert or round anything.
"""


class AnthropicCellNormalizer:
    """Rewrites refused cells with the model the project is configured for.

    Built lazily so constructing the importer costs nothing and needs no API
    key; the first refusal pays the setup. Effort is pinned low — this is
    string formatting, and the strict re-parse downstream is the quality gate,
    not the model's diligence.
    """

    def __init__(self, config: AnthropicConfig | None = None) -> None:
        self._config = config
        self._agent = None

    def _build(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
        from pydantic_ai.providers.anthropic import AnthropicProvider

        settings = self._config or AnthropicConfig()  # ty: ignore[missing-argument]
        return Agent(
            AnthropicModel(
                settings.default_model,
                provider=AnthropicProvider(api_key=settings.api_key),
            ),
            name="normalize_poll_cell",
            instructions=_INSTRUCTIONS,
            output_type=Rewrite,
            model_settings=AnthropicModelSettings(anthropic_effort="low"),
        )

    def rewrite(self, kind: str, text: str) -> str | None:
        form = CANONICAL_FORMS.get(kind)
        if form is None:
            return None
        if self._agent is None:
            self._agent = self._build()
        result = self._agent.run_sync(
            f"Cell kind: {kind}. Target form: {form}.\nCell text: {text!r}"
        )
        canonical = result.output.canonical
        logger.info("normalized %s cell %r -> %r", kind, text, canonical)
        return canonical
