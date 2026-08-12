"""Polls: the scraped shape, identity, and the deduplicating write path.

Polls do not go through the claim tables — they have nine tables of their own —
but identity works the same way as everywhere else in this project: derive it,
never read it from a name. Three layers, outermost first:

- **`PollKey`** (pollster + contest + fieldwork end) decides which *poll* this
  is, so Wikipedia and Ballotpedia reporting the same survey land on one `Poll`
  row instead of two.
- **`payload_hash`** decides whether this *reading* of the poll is new. The
  payload is the interpretation — numbers, dates, sample — and deliberately not
  the citation, so a second outlet reporting identical numbers is a no-op while
  a different reading of the same poll becomes a second revision plus a
  `ReviewTask`: two sources disagreeing about one poll is exactly what review
  is for.
- **Fuzzy checks** never merge, they only flag. A trigram near-miss on the
  pollster name, or another poll of the same race by the same pollster ending
  within a few days, files a `ReviewTask` and proceeds. Wrongly merging
  attributes one pollster's work to another silently; a fork or a duplicate is
  visible and repairable.

Candidate columns are stored as verbatim option labels with
`choice_entity_id` left NULL. Resolving "Rogers" to a person is a job for
whoever knows the contest's candidates (`candidate_in` exists for it); guessing
here would mint a PERSON named "Rogers", which is worse than not linking.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from decimal import Decimal
from typing import Literal
import uuid

from pydantic import Field
from sqlalchemy import func, select

from predictelection.research.contests import ContestKey, PollKey, normalize_slug
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.scraped import ScrapedEntity, ScrapedModel, ScrapedRecord
from predictelection.sql import (
    Entity,
    EntityAlias,
    EntityKind,
    Poll,
    PollRevision,
    ReviewTask,
    get_or_create,
    new_entity_alias,
    new_poll_estimate,
    new_poll_option,
    new_poll_revision,
)
from predictelection.sql.polling import PollQuestion, PollSample


POLL_KEY_NAMESPACE = "poll-key"
"""Written to Poll.external_namespace. Not in the identifier_namespace registry
because Poll's external identity columns are free text scoped to the poll
tables, not entity identifiers — the registry governs `entity_identifier`."""

POLLSTER_SIMILARITY_FLOOR = 0.55
"""Trigram similarity above which an unmatched pollster name is worth a look.

Low deliberately: this feeds a ReviewTask, not a merge, so a false positive
costs a reviewer seconds while a false negative is a silent fork.
"""

NEAR_DUPLICATE_WINDOW = dt.timedelta(days=3)
"""Same pollster, same contest, fieldwork ending within this of an existing
poll, but a different key — usually two sources rounding the field dates
differently, which is a duplicate wearing a disguise."""


class PollReading(ScrapedModel):
    """One option's number, exactly as the source printed it."""

    label: str = Field(
        min_length=1,
        max_length=300,
        description="The column or row label, verbatim — 'Stevens', 'Undecided'.",
    )
    percentage: Decimal | None = Field(
        default=None,
        ge=0,
        le=100,
        description="The printed percentage. Null when the source gave a count.",
    )
    response_count: int | None = Field(default=None, ge=0)


class ScrapedPoll(ScrapedRecord):
    """One poll as a source reported it.

    `contest` must carry a `contest_key` — the importer or workflow derives it
    from what it knows about the race; a poll that cannot be attached to a
    contest has nothing to correlate against and should not be ingested.
    """

    record_type: Literal["poll"] = "poll"

    pollster: str = Field(
        min_length=1,
        max_length=300,
        description="Who conducted it, as printed — 'EPIC-MRA', 'Emerson College'.",
    )
    sponsor: str | None = Field(
        default=None,
        max_length=300,
        description="Who paid for it, when the source says.",
    )
    contest: ScrapedEntity = Field(
        description="The race polled. Must carry contest_key."
    )
    fieldwork_started_on: dt.date | None = None
    fieldwork_ended_on: dt.date | None = Field(
        default=None,
        description=(
            "Last day in the field. This dates the poll itself — without it the "
            "poll cannot be deduplicated across sources, only within one."
        ),
    )
    published_on: dt.date | None = None
    sample_size: int | None = Field(default=None, gt=0)
    margin_of_error: Decimal | None = Field(default=None, ge=0)
    population: str = Field(
        default="unknown",
        max_length=100,
        description="Who was sampled: 'lv' likely voters, 'rv' registered, 'a' adults.",
    )
    readings: tuple[PollReading, ...] = Field(min_length=1)

    def payload(self) -> dict[str, object]:
        """The interpretation, for content dedup. Excludes the citation.

        Two outlets printing the same numbers must hash identically, so the
        source_url stays out; it lives on the revision's snapshot, which is
        provenance rather than identity.
        """

        return self.model_dump(mode="json", exclude={"source_url"})


@dataclass(frozen=True, slots=True)
class ResolvedPollster:
    entity_id: uuid.UUID
    created: bool
    lookalikes: tuple[tuple[uuid.UUID, str], ...] = ()
    """Existing organizations whose names are suspiciously close. Never merged
    automatically; surfaced so ingest can file a ReviewTask."""


def resolve_pollster(context: IngestContext, name: str) -> ResolvedPollster:
    """Name to ORGANIZATION entity, matching on slug so punctuation collapses.

    `normalize_entity_name` folds case and whitespace only, so "EPIC-MRA" and
    "EPIC MRA" are different alias keys. Pollster names vary mostly in
    punctuation, so the slug ("epic-mra" for both) is recorded as an alias and
    matched first. Anything less exact than the slug goes through trigram
    similarity into a ReviewTask — never into a merge.
    """

    session = context.session
    slug = normalize_slug(name)

    match = session.execute(
        select(EntityAlias.entity_id)
        .join(Entity, Entity.id == EntityAlias.entity_id)
        .where(
            EntityAlias.normalized_name == slug,
            Entity.kind == EntityKind.ORGANIZATION,
        )
        .limit(1)
    ).scalar()
    if match is not None:
        # Known pollster under new punctuation: remember this spelling too. The
        # key must use the same normalization the alias row stores, or a miss
        # here becomes an IntegrityError whose recovery re-read also misses.
        alias = new_entity_alias(entity_id=match, name=name)
        get_or_create(
            session,
            alias,
            key=(EntityAlias.entity_id == match)
            & (EntityAlias.normalized_name == alias.normalized_name),
        )
        return ResolvedPollster(entity_id=match, created=False)

    lookalikes = tuple(
        (entity_id, alias)
        for entity_id, alias in session.execute(
            select(EntityAlias.entity_id, EntityAlias.name)
            .join(Entity, Entity.id == EntityAlias.entity_id)
            .where(
                Entity.kind == EntityKind.ORGANIZATION,
                func.similarity(EntityAlias.normalized_name, slug)
                >= POLLSTER_SIMILARITY_FLOOR,
            )
            .order_by(func.similarity(EntityAlias.normalized_name, slug).desc())
            .limit(5)
        )
    )

    resolved = context.resolve(EntityKind.ORGANIZATION, ScrapedEntity(name=name))
    if resolved.created:
        # The slug alias is what makes the next punctuation variant hit tier 1.
        get_or_create(
            session,
            new_entity_alias(entity_id=resolved.entity_id, name=slug),
            key=(EntityAlias.entity_id == resolved.entity_id)
            & (EntityAlias.normalized_name == slug),
        )
    return ResolvedPollster(
        entity_id=resolved.entity_id,
        created=resolved.created,
        lookalikes=lookalikes,
    )


def ingest_poll(poll: ScrapedPoll, context: IngestContext) -> Ingestion:
    """One poll into the poll tables, at most once per identity and content."""

    session = context.session
    if not poll.contest.contest_key:
        raise ValueError("a poll must name its contest by contest_key")
    contest_key = ContestKey.parse(poll.contest.contest_key)

    pollster = resolve_pollster(context, poll.pollster)
    contest = context.resolve(EntityKind.CONTEST, poll.contest)
    sponsor_id = (
        resolve_pollster(context, poll.sponsor).entity_id if poll.sponsor else None
    )

    key = (
        PollKey.build(
            contest=contest_key,
            pollster=poll.pollster,
            fieldwork_end=poll.fieldwork_ended_on,
        )
        if poll.fieldwork_ended_on is not None
        else None
    )

    if key is not None:
        poll_row, _ = get_or_create(
            session,
            Poll(external_namespace=POLL_KEY_NAMESPACE, external_id=str(key)),
            key=(Poll.external_namespace == POLL_KEY_NAMESPACE)
            & (Poll.external_id == str(key)),
        )
    else:
        # No fieldwork end, no identity: this Poll row is reachable only by
        # content, so cross-source dedup cannot work. Flagged below.
        poll_row = Poll()
        session.add(poll_row)
    session.flush()

    prior = list(
        session.scalars(select(PollRevision).where(PollRevision.poll_id == poll_row.id))
    )
    revision = new_poll_revision(
        payload=poll.payload(),
        poll_id=poll_row.id,
        revision_number=len(prior) + 1,
        source_snapshot_id=context.snapshot.id,
        research_run_id=context.research_run_id,
        origin=context.origin,
        created_by=context.asserted_by,
        pollster_id=pollster.entity_id,
        sponsor_id=sponsor_id,
        fieldwork_started_on=poll.fieldwork_started_on,
        fieldwork_ended_on=poll.fieldwork_ended_on,
        collection_mode=None,
    )
    revision, created = get_or_create(
        session,
        revision,
        key=(PollRevision.poll_id == poll_row.id)
        & (PollRevision.payload_hash == revision.payload_hash),
    )

    if created:
        _write_readings(session, revision, contest_id=contest.entity_id, poll=poll)
        for reason in _concerns(
            session, poll=poll, key=key, pollster=pollster, prior=prior
        ):
            session.add(ReviewTask(poll_revision=revision, reason=reason))
    session.flush()

    return Ingestion(
        subject_entity_id=contest.entity_id,
        recorded=(),
        subject_created=created,
        related_entity_ids=(pollster.entity_id,),
    )


def _write_readings(
    session, revision: PollRevision, *, contest_id: uuid.UUID, poll: ScrapedPoll
) -> None:
    sample = PollSample(
        poll_revision_id=revision.id,
        position=0,
        label="overall",
        population=poll.population,
        sample_size=poll.sample_size,
        margin_of_error=poll.margin_of_error,
    )
    question = PollQuestion(
        poll_revision_id=revision.id,
        contest_id=contest_id,
        position=0,
        text=f"{poll.contest.name} — voter preference",
    )
    session.add_all([sample, question])
    session.flush()

    for position, reading in enumerate(poll.readings):
        option = new_poll_option(
            question=question, position=position, label=reading.label
        )
        session.add(option)
        session.flush()
        session.add(
            new_poll_estimate(
                option=option,
                sample=sample,
                percentage=reading.percentage,
                response_count=reading.response_count,
            )
        )


def _concerns(
    session,
    *,
    poll: ScrapedPoll,
    key: PollKey | None,
    pollster: ResolvedPollster,
    prior: list[PollRevision],
) -> list[str]:
    """Everything about this ingestion a human should glance at."""

    concerns: list[str] = []

    if key is None:
        concerns.append(
            "poll has no fieldwork end date, so it could not be keyed; "
            "re-imports and other sources will not deduplicate against it"
        )

    if prior:
        concerns.append(
            f"a different reading of this poll already exists "
            f"(revision {len(prior)}); two sources disagree about its contents"
        )

    if pollster.created and pollster.lookalikes:
        closest = ", ".join(f"{name!r}" for _, name in pollster.lookalikes[:3])
        concerns.append(
            f"new pollster {poll.pollster!r} resembles existing: {closest}; "
            "merge via redirect if they are the same organization"
        )

    if key is not None:
        window_start = key.fieldwork_end - NEAR_DUPLICATE_WINDOW
        window_end = key.fieldwork_end + NEAR_DUPLICATE_WINDOW
        neighbours = session.execute(
            select(Poll.external_id)
            .join(PollRevision, PollRevision.poll_id == Poll.id)
            .where(
                Poll.external_namespace == POLL_KEY_NAMESPACE,
                Poll.external_id != str(key),
                Poll.external_id.startswith(f"{key.contest}/{key.pollster}/"),
                PollRevision.fieldwork_ended_on.between(window_start, window_end),
            )
            .limit(3)
        ).scalars()
        for neighbour in neighbours:
            concerns.append(
                f"possible duplicate: {neighbour} is the same pollster and "
                "contest with fieldwork ending within "
                f"{NEAR_DUPLICATE_WINDOW.days} days — sources may state the "
                "field dates differently"
            )

    return concerns
