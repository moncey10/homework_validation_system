from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Keep SQLite for now (works locally + for demo)
DATABASE_URL = "sqlite:///./homework.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
