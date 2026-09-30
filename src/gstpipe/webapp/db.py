"""Storage for the review application.

The audit trail is the point: for every field the database keeps what the model read, what a human
changed it to, and what was finally approved. A financial system has to be able to answer "where
did this number come from", not just "what is it".

SQLite keeps the demo self-contained; the schema is plain SQLAlchemy and moves to PostgreSQL by
changing the URL.
"""

from __future__ import annotations

from datetime import datetime, UTC
from decimal import Decimal
from pathlib import Path

from sqlalchemy import (Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String, Text,
                        create_engine, func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

MONEY = Numeric(14, 2)


class Base(DeclarativeBase):
    pass


class Business(Base):
    __tablename__ = "businesses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    sector: Mapped[str] = mapped_column(String(50))
    state_code: Mapped[str] = mapped_column(String(2))
    gstin: Mapped[str] = mapped_column(String(15))
    invoices: Mapped[list[Invoice]] = relationship(back_populates="business")


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    supplier_name: Mapped[str | None] = mapped_column(String(200))
    supplier_gstin: Mapped[str | None] = mapped_column(String(20))
    invoice_number: Mapped[str | None] = mapped_column(String(60))
    invoice_date: Mapped[Date | None] = mapped_column(Date)
    due_date: Mapped[Date | None] = mapped_column(Date)
    place_of_supply: Mapped[str | None] = mapped_column(String(4))
    taxable_total: Mapped[Decimal | None] = mapped_column(MONEY)
    cgst_total: Mapped[Decimal | None] = mapped_column(MONEY)
    sgst_total: Mapped[Decimal | None] = mapped_column(MONEY)
    igst_total: Mapped[Decimal | None] = mapped_column(MONEY)
    invoice_total: Mapped[Decimal | None] = mapped_column(MONEY)
    category: Mapped[str | None] = mapped_column(String(120))
    document_path: Mapped[str | None] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(20), default="needs_review", index=True)
    extraction_confidence: Mapped[float] = mapped_column(default=0.0)

    business: Mapped[Business] = relationship(back_populates="invoices")
    fields: Mapped[list[FieldValue]] = relationship(back_populates="invoice",
                                                      cascade="all, delete-orphan")
    flags: Mapped[list[Flag]] = relationship(back_populates="invoice",
                                               cascade="all, delete-orphan")


class FieldValue(Base):
    """One field of one invoice: what the model read, and what a human approved."""

    __tablename__ = "field_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), index=True)
    field: Mapped[str] = mapped_column(String(40))
    model_value: Mapped[str | None] = mapped_column(Text)
    approved_value: Mapped[str | None] = mapped_column(Text)
    corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    # Where on the page this was read from, as JSON, so the scan can be annotated.
    region: Mapped[str | None] = mapped_column(Text)
    invoice: Mapped[Invoice] = relationship(back_populates="fields")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), index=True)
    field: Mapped[str | None] = mapped_column(String(40))
    action: Mapped[str] = mapped_column(String(30))
    old_value: Mapped[str | None] = mapped_column(Text)
    new_value: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(60), default="ca@example.com")
    at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class Flag(Base):
    __tablename__ = "flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    detail: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[float] = mapped_column(default=0.0)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    invoice: Mapped[Invoice] = relationship(back_populates="flags")


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.id"), index=True)
    txn_date: Mapped[Date] = mapped_column(Date)
    amount: Mapped[Decimal] = mapped_column(MONEY)
    narration: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(30))
    matched_invoice_id: Mapped[str | None] = mapped_column(String(32), index=True)
    match_score: Mapped[float | None] = mapped_column()
    match_status: Mapped[str] = mapped_column(String(20), default="unmatched", index=True)


def make_engine(path: Path | str = "data/review.db"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path.as_posix()}", future=True)


def create_all(engine) -> None:
    Base.metadata.create_all(engine)


def reset(engine) -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


__all__ = ["Base", "Business", "Invoice", "FieldValue", "AuditEvent", "Flag", "Payment",
           "make_engine", "create_all", "reset", "Session", "select", "func"]
