from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from db import Base

class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(100), index=True, nullable=False)
    homework_id = Column(String(100), index=True, nullable=False)
    status = Column(String(50), default="processed")

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)

    filename = Column(String(255), nullable=False)
    extracted_text = Column(Text, nullable=True)
    segmented_answers_json = Column(Text, nullable=True)
    validation_json = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, nullable=True)
    level = Column(String(20), default="INFO")
    message = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
