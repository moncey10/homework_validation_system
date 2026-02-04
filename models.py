from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(100), unique=True, index=True, nullable=False)  # like "st01"
    name = Column(String(200), nullable=True)
    email = Column(String(200), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class HomeworkAssignment(Base):
    __tablename__ = "homework_assignments"

    id = Column(Integer, primary_key=True, index=True)
    homework_id = Column(String(100), unique=True, index=True, nullable=False)  # like "hw01"
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Submission(Base):
    """
    Keep your existing columns (student_id/homework_id as strings) so your current code works.
    Also add optional FK links for a more proper schema.
    """
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)

    # Existing fields used by your current API:
    student_id = Column(String(100), index=True, nullable=False)
    homework_id = Column(String(100), index=True, nullable=False)

    # Optional normalized references (can be filled later; not required now)
    student_ref_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    homework_ref_id = Column(Integer, ForeignKey("homework_assignments.id"), nullable=True)

    status = Column(String(50), default="processed")  # processed/failed/etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships (optional usage)
    student = relationship("Student", lazy="joined")
    homework = relationship("HomeworkAssignment", lazy="joined")


class HomeworkImage(Base):
    """
    Store each uploaded image record (required by plan).
    We store filename + content_type + optional disk_path.
    If you later want, you can store image bytes too (BLOB), but not needed now.
    """
    __tablename__ = "homework_images"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)

    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)
    disk_path = Column(String(500), nullable=True)  # if you save file to disk later

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)

    # Keep your existing filename field (ties to image filename)
    filename = Column(String(255), nullable=False)

    extracted_text = Column(Text, nullable=True)
    segmented_answers_json = Column(Text, nullable=True)
    validation_json = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # Prevent duplicate results for same submission+filename
        UniqueConstraint("submission_id", "filename", name="uq_result_submission_filename"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=True)

    level = Column(String(20), default="INFO")  # INFO/ERROR
    message = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
