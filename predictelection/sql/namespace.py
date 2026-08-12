"""The registry of external identifier schemes.

Entities carry identifiers from many schemes at once — Wikidata, OCD, FEC — and
practice is that all of them keep existing rather than one winning. Schemes also
retire: eduPersonTargetedID was marked deprecated with a recommended successor
and later made obsolete, and OCD could go the same way inside this project's
lifetime.

So a scheme is a row, not a string literal. Deprecating one is an update here,
touching no identifier. Nothing in resolution may hardcode a namespace; which
scheme wins a disagreement is `precedence`, which is data.

The registry also makes the namespace column typo-proof. `EntityIdentifier`
previously took free text guarded only by a lowercase CHECK, so `wikdata` was as
valid as `wikidata` and would simply never match anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from predictelection.sql.base import Base, created_at_timestamp, enum_type
from predictelection.sql.entity import normalize_identifier_namespace


class NamespaceStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    """Still resolvable, no longer issued. Prefer `superseded_by`."""

    OBSOLETE = "obsolete"
    """Retained only so historical identifiers still resolve."""


class IdentifierNamespace(Base):
    """One external identifier scheme.

    Mutable on purpose, unlike most of this schema: a scheme's status is a fact
    about the outside world that genuinely changes, and marking one deprecated
    should not require rewriting the identifiers that use it.
    """

    __tablename__ = "identifier_namespace"

    namespace: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(255))
    authority: Mapped[str | None] = mapped_column(String(255))
    """Who issues these, when anyone does."""

    uri_template: Mapped[str | None] = mapped_column(Text)
    """e.g. https://www.wikidata.org/wiki/{value} — for citing, not fetching."""

    status: Mapped[NamespaceStatus] = mapped_column(
        enum_type(NamespaceStatus, name="identifier_namespace_status"),
        default=NamespaceStatus.ACTIVE,
    )
    superseded_by: Mapped[str | None] = mapped_column(
        ForeignKey("identifier_namespace.namespace", ondelete="RESTRICT")
    )
    precedence: Mapped[int] = mapped_column(Integer, default=100)
    """Lower wins when two identifiers disagree about which entity this is."""

    created_at: Mapped[created_at_timestamp]

    __table_args__ = (
        CheckConstraint(
            "namespace = lower(btrim(namespace)) AND namespace <> ''",
            name="namespace_normalized",
        ),
        CheckConstraint("precedence >= 0", name="precedence_nonnegative"),
        CheckConstraint(
            "superseded_by IS NULL OR superseded_by <> namespace",
            name="does_not_supersede_self",
        ),
        CheckConstraint(
            """
            status = 'active' OR superseded_by IS NOT NULL
            OR status = 'obsolete'
            """,
            name="deprecated_names_a_successor",
        ),
    )


@dataclass(frozen=True, slots=True)
class NamespaceSpec:
    """The Python source of truth for one scheme, seeded like PredicateSpec."""

    namespace: str
    label: str
    precedence: int
    authority: str | None = None
    uri_template: str | None = None

    def __post_init__(self) -> None:
        if self.namespace != normalize_identifier_namespace(self.namespace):
            raise ValueError(f"namespace must be normalized: {self.namespace!r}")
        if self.precedence < 0:
            raise ValueError("precedence must not be negative")


NAMESPACE_SPECS: tuple[NamespaceSpec, ...] = (
    NamespaceSpec(
        namespace="ocd-division",
        label="Open Civic Data division",
        authority="Open Civic Data",
        # Beats Wikidata for US geography specifically: it is what public
        # election datasets key on, so imports line up without reconciliation.
        precedence=5,
    ),
    NamespaceSpec(
        namespace="wikidata",
        label="Wikidata QID",
        authority="Wikimedia Foundation",
        uri_template="https://www.wikidata.org/wiki/{value}",
        # Broadest coverage of people and organizations, and the most stable
        # cross-domain identifier available.
        precedence=10,
    ),
    NamespaceSpec(
        namespace="fec",
        label="FEC committee or candidate ID",
        authority="US Federal Election Commission",
        uri_template="https://www.fec.gov/data/candidate/{value}/",
        precedence=20,
    ),
    NamespaceSpec(
        namespace="bioguide",
        label="Congressional Biographical Directory ID",
        authority="US Congress",
        uri_template="https://bioguide.congress.gov/search/bio/{value}",
        precedence=20,
    ),
    NamespaceSpec(
        namespace="kalshi",
        label="Kalshi market ticker",
        authority="Kalshi",
        precedence=30,
    ),
    NamespaceSpec(
        namespace="ballotpedia",
        label="Ballotpedia page slug",
        authority="Ballotpedia",
        uri_template="https://ballotpedia.org/{value}",
        # Editorially maintained and re-slugged from time to time, so it loses
        # to anything issued by an authority.
        precedence=50,
    ),
    NamespaceSpec(
        namespace="contest-key",
        label="Derived contest key",
        # Issued by this project, because nobody issues one. A contest has no
        # external identifier, so three sources describing the same race would
        # otherwise fork it three ways on wording alone. The value is derived
        # from division, office, cycle, stage and party — see
        # research/contests.py — so anything that can describe a contest
        # arrives at the same string without coordination.
        authority=None,
        # Last: derived by us from other people's facts, so any identifier an
        # actual authority issued should win a disagreement.
        precedence=60,
    ),
    NamespaceSpec(
        namespace="office-key",
        label="Derived office key",
        # Same argument as contest-key, for the seat rather than the race:
        # division plus office, with no cycle, because a seat outlives any one
        # election. Without it "every governorship up in 2026" cannot be asked.
        authority=None,
        precedence=60,
    ),
    NamespaceSpec(
        namespace="election-key",
        label="Derived election key",
        # Division, cycle and stage. Several contests share one election day,
        # and that grouping is only reliable if everyone derives the same ID
        # for it rather than naming it.
        authority=None,
        precedence=60,
    ),
    NamespaceSpec(
        namespace="event-key",
        label="Derived event key",
        # Division, kind and date, plus a host when two events share a day.
        # This is the identity that was actually measured breaking: two runs on
        # one subject produced 11 event entities for 6 real debates purely from
        # re-phrased titles. A date does not get re-phrased.
        authority=None,
        precedence=60,
    ),
)


def seed_identifier_namespaces(session: Session) -> None:
    """Insert or refresh the registry. Idempotent, like seed_predicates."""

    existing = {
        row.namespace: row for row in session.scalars(select(IdentifierNamespace))
    }
    for spec in NAMESPACE_SPECS:
        current = existing.get(spec.namespace)
        if current is None:
            session.add(
                IdentifierNamespace(
                    namespace=spec.namespace,
                    label=spec.label,
                    authority=spec.authority,
                    uri_template=spec.uri_template,
                    precedence=spec.precedence,
                )
            )
            continue
        # Descriptive fields may improve; status and superseded_by are
        # operational decisions and are deliberately left alone once set.
        current.label = spec.label
        current.authority = spec.authority
        current.uri_template = spec.uri_template
        current.precedence = spec.precedence
    session.flush()


def active_namespaces(session: Session) -> list[IdentifierNamespace]:
    """Schemes worth writing new identifiers into, best first."""

    return list(
        session.scalars(
            select(IdentifierNamespace)
            .where(IdentifierNamespace.status == NamespaceStatus.ACTIVE)
            .order_by(IdentifierNamespace.precedence)
        )
    )
