from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# Locate the project root.
BASE_DIR = Path(__file__).resolve().parents[2]

# Store our SQLite database in the project root.
DATABASE_PATH = BASE_DIR / "sql_agent.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"


# Create the SQLAlchemy engine.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)


# Enable foreign-key enforcement for every SQLite connection.
@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Create a reusable factory for database sessions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


# Parent class for all database models.
class Base(DeclarativeBase):
    pass