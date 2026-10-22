"""Programmatic Alembic migrations (run automatically by the API and worker on startup)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from .config import get_settings

MIGRATIONS = Path(__file__).with_name("migrations")


def alembic_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", get_settings().resolved_database_url)
    return cfg


def upgrade(revision: str = "head") -> None:
    get_settings().data_dir.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(), revision)


def revision(message: str) -> None:
    command.revision(alembic_config(), message=message, autogenerate=True)
