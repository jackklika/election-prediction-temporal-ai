from predictelection.sql.base import Base


def test_create_tables():
    """Test basic creation of all tables"""
    from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient

    client = SqlAlchemyEngineClient()
    Base.metadata.create_all(
        client.engine
    )
