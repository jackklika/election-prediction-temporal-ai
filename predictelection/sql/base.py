from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum as PythonEnum
from enum import StrEnum
from hashlib import sha256
import json
from typing import Annotated, Any
import uuid

from pydantic import AfterValidator
from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum as SqlEnum,
    Identity,
    MetaData,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, mapped_column, object_session


_NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

uuid_primary_key = Annotated[
    uuid.UUID,
    mapped_column(primary_key=True, default=uuid.uuid4),
]
utc_timestamp = Annotated[datetime, mapped_column(DateTime(timezone=True))]
created_at_timestamp = Annotated[
    datetime,
    mapped_column(DateTime(timezone=True), server_default=func.now()),
]
insert_sequence = Annotated[
    int,
    mapped_column(BigInteger, Identity(always=True)),
]
"""A monotonic insert counter for rows whose current state is "the latest one".

created_at uses now(), which PostgreSQL evaluates once per transaction, so rows
written by a single research run share a timestamp and cannot be ordered. This
column breaks those ties. Note it records assignment order, not commit order:
under concurrency a lower sequence may become visible after a higher one, so
this is a strong ordering rather than a serialization point.
"""


class TimePrecision(StrEnum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    EXACT = "exact"


class RecordOrigin(StrEnum):
    MODEL = "model"
    HUMAN = "human"
    IMPORT = "import"
    SYSTEM = "system"


ENUM_VALUE_LENGTH = 32
"""Fixed VARCHAR width for every checked enum column.

Without this, SQLAlchemy sizes the column to the longest current member, so
adding a longer member later would need ALTER COLUMN TYPE on top of rebuilding
the CHECK constraint. A fixed width keeps enum growth to a CHECK swap.
"""


def nullable_jsonb() -> JSONB:
    """JSONB where a Python None means SQL NULL, not the JSON value null.

    SQLAlchemy's default is the opposite: None is persisted as 'null'::jsonb,
    which is not SQL NULL, so every constraint phrased as "value IS NULL" is
    silently false and rejects the row it was meant to allow. Every nullable
    JSONB column in this package must use this.
    """

    return JSONB(none_as_null=True)


def enum_type(enum_class: type[PythonEnum], *, name: str) -> SqlEnum:
    """Build a checked VARCHAR enum that persists each member's value."""

    longest_value = max(len(str(member.value)) for member in enum_class)
    if longest_value > ENUM_VALUE_LENGTH:
        raise ValueError(
            f"{enum_class.__name__} has a member longer than {ENUM_VALUE_LENGTH}"
        )

    return SqlEnum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        length=ENUM_VALUE_LENGTH,
        values_callable=lambda members: [str(member.value) for member in members],
    )


def canonical_decimal(value: Decimal) -> str:
    """Render a Decimal so equal numbers always produce equal text.

    str() keeps the original scale, so Decimal("12.5") and Decimal("12.50")
    would hash differently and defeat the fingerprint unique constraints.
    normalize() strips trailing zeros, and the "f" format avoids the
    scientific notation normalize() otherwise introduces for integers.
    """

    if not value.is_finite():
        raise ValueError("canonical decimals must be finite")
    return format(value.normalize(), "f")


def _canonicalize_decimal(value: Decimal) -> Decimal:
    return Decimal(canonical_decimal(value))


CanonicalDecimal = Annotated[Decimal, AfterValidator(_canonicalize_decimal)]
"""A Decimal that discards scale on validation.

Pydantic serializes Decimal to a scale-preserving string in JSON mode, so a
plain Decimal field reaches canonical_json already stringified as "12.50" and
_canonical_json_default never sees it. Value models and evidence locators whose
numbers participate in a fingerprint must use this type instead.
"""


def _canonical_json_default(value: object) -> str:
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("canonical datetimes must be timezone-aware")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, PythonEnum):
        return str(value.value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for identities and audit hashes."""

    return json.dumps(
        value,
        default=_canonical_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def idempotency_key(operation: str, /, **parts: Any) -> str:
    """Build a key that is stable across retries of the same logical write.

    Every idempotency_key column is unique, which only helps if the key is a
    function of *what is being written* rather than of when. A retried Temporal
    activity must reproduce the key exactly so the second attempt is rejected;
    a uuid4 or a timestamp would defeat the column entirely.

    Deliberately not derived from Temporal identifiers: ResearchRun is documented
    as working whether or not Temporal ran it, and uq_research_run_temporal_execution
    already covers the workflow-execution angle. Hashing content works for both.

    The exception is ReviewDecision. Human review is interactive rather than
    retried, and a reviewer must be able to change their mind later, since the
    latest decision wins. Pass a per-action token from the UI there instead of
    hashing the outcome.

    The readable prefix is for debugging: a unique violation then names the
    operation that collided instead of showing an opaque digest.
    """

    if not operation or operation != operation.strip():
        raise ValueError("idempotency operations need a non-blank name")
    return f"{operation}:{canonical_json_sha256(parts)}"


class Base(DeclarativeBase):
    """Base for all ORM models."""

    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


class Immutable:
    """Marker for append-only rows.

    The ORM guards below prevent ordinary Session updates and deletes. Database
    roles should also deny UPDATE and DELETE in deployments where tamper
    resistance is required; SQLAlchemy events cannot protect direct SQL.
    """


@event.listens_for(Base, "before_update", propagate=True)
def _prevent_immutable_update(
    mapper: object, connection: object, target: object
) -> None:
    del mapper, connection
    if not isinstance(target, Immutable):
        return

    session = object_session(target)
    if session is None or session.is_modified(target, include_collections=False):
        raise TypeError(f"{type(target).__name__} rows are immutable")


@event.listens_for(Base, "before_delete", propagate=True)
def _prevent_immutable_delete(
    mapper: object, connection: object, target: object
) -> None:
    del mapper, connection
    if isinstance(target, Immutable):
        raise TypeError(f"{type(target).__name__} rows are immutable")
