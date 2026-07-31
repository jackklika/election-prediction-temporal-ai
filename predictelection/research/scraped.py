"""The vocabulary every scraper shares, independent of what it scrapes.

These models describe what an agent *observed*, in the agent's terms: names, not
IDs, because a scraper has no way to know an ID. Resolution to entities happens
during ingestion.

They double as the agents' output contracts, so the field descriptions are the
prompt. Keeping them here rather than beside the first domain that needed them
means a new domain inherits the rules instead of restating them — and cannot
quietly omit one.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from predictelection.sql import EntityKind, EntityMention, ExternalIdentifier


class ScrapedModel(BaseModel):
    """Base for everything in this layer.

    extra="forbid" is the load-bearing part: a model that invents a field should
    fail validation loudly rather than have it silently dropped, because a
    dropped field looks identical to one the source never mentioned.
    """

    model_config = ConfigDict(extra="forbid")


class ScrapedEntity(ScrapedModel):
    """A named thing an agent saw, with an external ID when the source gave one.

    Deliberately not a ScrapedRecord: a person named inside a debate has no
    citation of its own, it inherits the containing record's. Only things
    reported as facts have to cite.
    """

    name: str = Field(
        min_length=1,
        max_length=500,
        description="The name exactly as the source writes it, not an abbreviation.",
    )
    wikidata_id: str | None = Field(
        default=None,
        pattern=r"^Q[0-9]+$",
        description=(
            "Wikidata QID such as Q28137641, only if you are certain it is the "
            "right one. It resolves identity exactly, so a wrong QID merges two "
            "different people. Leave null when unsure."
        ),
    )
    ocd_id: str | None = Field(
        default=None,
        pattern=r"^ocd-(division|jurisdiction)/.+$",
        description=(
            "Open Civic Data ID such as ocd-division/country:us/state:wi. The "
            "best identifier for a US jurisdiction. Leave null when unsure."
        ),
    )
    fec_id: str | None = Field(
        default=None,
        max_length=32,
        description="FEC candidate or committee ID, for US federal races.",
    )

    _IDENTIFIER_FIELDS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("wikidata", "wikidata_id"),
        ("ocd-division", "ocd_id"),
        ("fec", "fec_id"),
    )
    """Field-to-namespace map. Named fields because a model fills in `ocd_id`
    far more reliably than a free-form namespace/value pair — but the mapping is
    a table rather than an if-chain, so adding a scheme is one row here and one
    row in the registry, and no resolution code names a namespace."""

    def as_mention(self, kind: EntityKind) -> EntityMention:
        """Hand this to resolve_entity_mention to get a stable entity ID.

        Every identifier the source offered is carried, not just the first.
        Schemes coexist and outlive each other; resolution decides which wins by
        the registry's precedence, so nothing here has to.
        """

        identifiers = tuple(
            ExternalIdentifier(namespace=namespace, value=value)
            for namespace, field in self._IDENTIFIER_FIELDS
            if (value := getattr(self, field))
        )
        return EntityMention(kind=kind, name=self.name, identifiers=identifiers)


class ScrapedRecord(ScrapedModel):
    """Base for anything an agent reports as a fact.

    source_url lives here rather than on each domain model because the
    requirement is universal and absolute: the page is archived, and every claim
    derived from this record cites that archive. A fact nobody can re-check is
    the one thing the provenance model exists to prevent, so a domain model that
    forgot to ask for a citation would quietly defeat it.
    """

    source_url: str = Field(
        min_length=1,
        description=(
            "The page you learned this from. Required: it is archived and every "
            "resulting fact cites it, so anything you cannot cite must be "
            "omitted rather than reported."
        ),
    )
