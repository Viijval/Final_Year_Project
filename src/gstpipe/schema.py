"""Ground-truth record types. Every field here is the *true* value; model outputs are compared against these."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Split = Literal["train", "val", "test"]


class LegalForm(str, Enum):
    proprietorship = "proprietorship"
    firm = "firm"
    llp = "llp"
    company = "company"


class Party(BaseModel):
    party_id: str
    name: str
    legal_form: LegalForm
    pan: str
    gstin: str
    state_code: str
    city: str
    address: str


class LineItem(BaseModel):
    line_no: int
    description: str
    hsn_sac: str
    quantity: Decimal
    unit_price: Decimal
    taxable_value: Decimal
    gst_rate: Decimal
    cgst: Decimal
    sgst: Decimal
    igst: Decimal
    category: str


class Invoice(BaseModel):
    invoice_id: str
    business_id: str
    split: Split
    invoice_number: str
    invoice_date: date
    due_date: date
    supplier: Party
    recipient: Party
    place_of_supply: str
    line_items: list[LineItem]
    taxable_total: Decimal
    cgst_total: Decimal
    sgst_total: Decimal
    igst_total: Decimal
    round_off: Decimal
    invoice_total: Decimal
    category: str = Field(description="Primary category: the leaf with the largest taxable value")


class TxnKind(str, Enum):
    invoice_payment = "invoice_payment"
    unbilled_vendor_payment = "unbilled_vendor_payment"
    salary = "salary"
    bank_charge = "bank_charge"
    gst_remittance = "gst_remittance"
    loan_emi = "loan_emi"


class Transaction(BaseModel):
    txn_id: str
    business_id: str
    split: Split
    txn_date: date
    amount: Decimal
    mode: Literal["NEFT", "RTGS", "IMPS", "UPI", "CHQ", "ACH"]
    narration: str
    kind: TxnKind
    counterparty_id: str | None = None


class MatchType(str, Enum):
    exact = "exact"
    net_of_tds = "net_of_tds"
    partial = "partial"
    consolidated = "consolidated"


class Match(BaseModel):
    txn_id: str
    invoice_id: str
    allocated_amount: Decimal
    tds_amount: Decimal
    match_type: MatchType


class AnomalyType(str, Enum):
    duplicate_invoice = "duplicate_invoice"
    near_duplicate = "near_duplicate"
    amount_outlier = "amount_outlier"
    gst_arithmetic_error = "gst_arithmetic_error"
    wrong_tax_type = "wrong_tax_type"
    invalid_gstin = "invalid_gstin"
    due_before_invoice_date = "due_before_invoice_date"
    unbilled_payment = "unbilled_payment"


class Anomaly(BaseModel):
    anomaly_id: str
    entity_type: Literal["invoice", "transaction"]
    entity_id: str
    anomaly_type: AnomalyType
    detail: str
