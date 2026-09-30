"""Whether a figure can be signed off, and what is standing in the way.

The research behind this system found that extraction errors turn into false alarms downstream: a
misread digit looks exactly like a genuine inconsistency. The conclusion for a product is blunt —
never present a number as settled unless you can show what it was read from and what was checked.

So every invoice sits in one of three states, and the reasons are always visible:

    verified      read cleanly, every check passed, or a person signed it off
    needs_check   usable, but something failed and a person should look
    unreadable    the essentials could not be read, so it is not counted anywhere
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

VERIFIED = "verified"
NEEDS_CHECK = "needs_check"
UNREADABLE = "unreadable"

STATE_LABEL = {VERIFIED: "Verified", NEEDS_CHECK: "Needs checking", UNREADABLE: "Could not read"}
STATE_TONE = {VERIFIED: "good", NEEDS_CHECK: "warn", UNREADABLE: "bad"}

# Without these an invoice cannot be used for anything: no total, no date, no identity.
ESSENTIAL = ("invoice_number", "invoice_date", "invoice_total")
IDENTITY = ("supplier_gstin", "supplier_name")
CONFIDENCE_FLOOR = 0.75
ARITHMETIC_TOLERANCE = Decimal("1.00")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str

    @property
    def tone(self) -> str:
        return "good" if self.ok else "warn"


@dataclass
class Verification:
    state: str
    checks: list[Check]

    @property
    def label(self) -> str:
        return STATE_LABEL[self.state]

    @property
    def tone(self) -> str:
        return STATE_TONE[self.state]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def score(self) -> float:
        return sum(c.ok for c in self.checks) / len(self.checks) if self.checks else 0.0

    @property
    def signed_off(self) -> bool:
        return self.state == VERIFIED


def _value(invoice, field: str):
    return getattr(invoice, field, None)


def assess(invoice, *, signed_off: bool = False, traced: set[str] | None = None) -> Verification:
    """Judge one invoice. `traced` is the set of fields whose source region is known."""
    traced = traced or set()
    open_flags = [f for f in getattr(invoice, "flags", []) if not f.resolved]

    missing = [f for f in ESSENTIAL if _value(invoice, f) is None]
    has_identity = any(_value(invoice, f) for f in IDENTITY)
    confidence = float(invoice.extraction_confidence or 0.0)

    checks = [
        Check("Essentials read", not missing,
              "number, date and total are all present" if not missing
              else "could not read: " + ", ".join(f.replace("_", " ") for f in missing)),
        Check("Supplier identified", has_identity,
              "supplier is named or has a GSTIN" if has_identity else "no supplier name or GSTIN"),
        Check("Traceable to the document", bool(traced),
              f"{len(traced)} values located on the page" if traced
              else "no values could be located on the page"),
        Check("Read confidently", confidence >= CONFIDENCE_FLOOR,
              f"{confidence:.0%} confidence" if confidence
              else "no confidence recorded"),
    ]

    taxable, total = _value(invoice, "taxable_total"), _value(invoice, "invoice_total")
    if taxable is not None and total is not None:
        tax = sum((_value(invoice, f) or Decimal("0")
                   for f in ("cgst_total", "sgst_total", "igst_total")), Decimal("0"))
        gap = abs(taxable + tax - total)
        checks.append(Check("Adds up", gap <= ARITHMETIC_TOLERANCE,
                            "taxable plus tax matches the total" if gap <= ARITHMETIC_TOLERANCE
                            else f"taxable plus tax is off the total by {gap:,.2f}"))
    else:
        checks.append(Check("Adds up", False, "not enough figures to check the arithmetic"))

    checks.append(Check("Nothing flagged", not open_flags,
                        "no checks failed" if not open_flags
                        else "; ".join(f.detail for f in open_flags[:3])))

    if missing or not has_identity:
        state = UNREADABLE
    elif signed_off:
        state = VERIFIED
    elif all(c.ok for c in checks):
        state = VERIFIED
    else:
        state = NEEDS_CHECK
    return Verification(state=state, checks=checks)


def summarise(rows: list[tuple[Verification, Decimal | None]]) -> dict:
    """Counts and value at stake in each state, for the overview."""
    out = {state: {"count": 0, "value": Decimal("0")} for state in (VERIFIED, NEEDS_CHECK, UNREADABLE)}
    for verification, amount in rows:
        bucket = out[verification.state]
        bucket["count"] += 1
        bucket["value"] += amount or Decimal("0")
    total = sum(b["count"] for b in out.values()) or 1
    for state, bucket in out.items():
        bucket["share"] = 100 * bucket["count"] / total
        bucket["label"] = STATE_LABEL[state]
        bucket["tone"] = STATE_TONE[state]
    return out
