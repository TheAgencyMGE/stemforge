"""Alembic environment; the URL comes from Stemforge settings, never from alembic.ini."""

from alembic import context

from stemforge import models  # noqa: F401 - register tables
from stemforge.db import Base, get_engine

target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = context.config.attributes.get("connection") or get_engine()
    if hasattr(connectable, "connect"):
        with connectable.connect() as connection:
            _run(connection)
    else:
        _run(connectable)


def _run(connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite", compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
