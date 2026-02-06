from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=True)
    email = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class HomeworkAssignment(Base):
    __tablename__ = "homework_assignments"

    id = Column(Integer, primary_key=True, index=True)
    homework_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(String(100), index=True, nullable=False)
    homework_id = Column(String(100), index=True, nullable=False)

    student_ref_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    homework_ref_id = Column(Integer, ForeignKey("homework_assignments.id"), nullable=True)

    status = Column(String(50), default="processed")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", lazy="joined")
    homework = relationship("HomeworkAssignment", lazy="joined")


class HomeworkImage(Base):
    __tablename__ = "homework_images"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=True)

    homework_id = Column(String(100), index=True, nullable=False)
    role = Column(String(20), nullable=False)  # teacher/student

    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)

    disk_path = Column(String(500), nullable=True)
    ocr_text = Column(Text, nullable=True)

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

    __table_args__ = (
        UniqueConstraint("submission_id", "filename", name="uq_result_submission_filename"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=True)

    level = Column(String(20), default="INFO")
    message = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
