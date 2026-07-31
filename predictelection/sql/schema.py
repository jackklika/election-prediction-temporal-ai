from sqlalchemy import Engine
from sqlalchemy.orm import Session

from predictelection.sql.base import Base
from predictelection.sql.predicate import seed_predicates


def create_schema(engine: Engine) -> None:
    """Create all registered tables and seed immutable predicate contracts."""

    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        seed_predicates(session)


def rebuild_schema(engine: Engine) -> None:
    """Drop the current model and recreate it; intended only before real data."""

    Base.metadata.drop_all(engine)
    create_schema(engine)
