from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, UniqueConstraint, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property


timestamp_utc = Annotated[
    datetime, mapped_column(DateTime(timezone=True), server_default=func.now())
]


class Base(DeclarativeBase):
    """Base ORM model"""


class Entity(Base):
    """A thing a fact can be about, 'Proper Noun'"""

    __tablename__: str = "entity"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[str]
    canonical_name: Mapped[str]  # The official name of this entity we agree on
    # aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list) # probably do via differnt table for efficiency
    wikidata_id: Mapped[
        str | None
    ]  # wikidata is used for our best "unique identifier" for entities

    # IDs from other sources that refer to this entity. This is more important than canonical_name since
    # there can be duplicates. We have wikidata as top-level column since it is present for most major
    # entities already, and is a good default.
    #
    # For example for Francesca Hong:
    # {"wikidata":"Q102181078", "wikipedia_curid":"65899862"}
    external_ids: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    created_at: Mapped[timestamp_utc]

    __table_args__ = (
        Index(
            "uq_entity_wikidata",
            "wikidata_id",
            unique=True,
            postgresql_where=text("wikidata_id IS NOT NULL"),
        ),
    )


class Source(Base):
    """Where a fact came from"""

    __tablename__: str = "source"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[str]
    url: Mapped[str | None]
    publisher: Mapped[str | None]

    created_at: Mapped[timestamp_utc]


class Fact(Base):
    __tablename__: str = "fact"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    predicate: Mapped[str]

    # What the fact is about
    subject_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entity.id"))

    # the other entity the fact points at when the predicate is a relationship between two entities, like "endorsed_by"
    object_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("entity.id"))
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source.id"))

    value: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[timestamp_utc]
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # when the fact is true at

    __mapper_args__: dict[str, str | bool] = {
        # Specifies the column, attribute, or SQL expression used to determine the target class
        # for an incoming row, when inheriting classes are present.
        # This is used so we can have orm models per fact type
        # https://docs.sqlalchemy.org/en/20/orm/mapping_api.html#sqlalchemy.orm.Mapper.params.polymorphic_on
        "polymorphic_on": "predicate",
        "polymorphic_abstract": True,
    }

    __table_args__: tuple = (
        # Helps us ensure true idempotent
        UniqueConstraint(
            "subject_id",
            "predicate",
            "source_id",
            "observed_at",
            name="uq_fact_idem",
            postgresql_nulls_not_distinct=True,  # ensure we don't factor null into distinct
        ),
    )

class Poll(Fact):
    __mapper_args__ = {"polymorphic_identity": "poll_average"}

    @hybrid_property
    def pct(self) -> Decimal:
        return self.value["pct"]

    @pct.inplace.expression
    @classmethod
    def _pct(cls):
        """
        SQL-side expression so we can reference this in sql queries

        If we do `value -> 'pct'` this gets a JSON value, not a number.
        So this fixes the extraction, treating it as a number.
        """
        return cls.value["pct"].astext.cast(Numeric)



if __name__ == "__main__":
    from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient

    client = SqlAlchemyEngineClient()
    Base.metadata.create_all(
        client.engine
    )  # convert this to migrations via Alembic later
