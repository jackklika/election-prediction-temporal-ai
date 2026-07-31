from sqlalchemy import Engine
from sqlalchemy.orm import Session

from predictelection.sql.base import Base
from predictelection.sql.namespace import seed_identifier_namespaces
from predictelection.sql.predicate import seed_predicates


def create_schema(engine: Engine) -> None:
    """Create all registered tables and seed immutable predicate contracts."""

    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        seed_identifier_namespaces(session)
        seed_predicates(session)
