"""Category taxonomy, purchasable item specs and vendor naming material.

HSN/SAC codes and rates are illustrative for synthetic data generation, not a tax reference.

Two properties are deliberate. Some HSN codes are shared between related categories (998313 covers
both cloud services and IT consulting), and descriptions reuse generic business vocabulary
("charges", "services", "monthly"), so categorisation cannot be solved by a lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

TAXONOMY: dict[str, tuple[str, ...]] = {
    "cloud_services":     ("Expense", "Operating", "Technology", "Cloud Services"),
    "software_licences":  ("Expense", "Operating", "Technology", "Software Licences"),
    "it_consulting":      ("Expense", "Operating", "Technology", "IT Consulting"),
    "it_hardware":        ("Expense", "Capital", "Equipment", "IT Hardware"),
    "furniture":          ("Expense", "Capital", "Equipment", "Furniture"),
    "rent":               ("Expense", "Operating", "Premises", "Rent"),
    "telecom":            ("Expense", "Operating", "Premises", "Internet & Telecom"),
    "repairs":            ("Expense", "Operating", "Premises", "Repairs & Maintenance"),
    "legal":              ("Expense", "Operating", "Professional Services", "Legal"),
    "accounting":         ("Expense", "Operating", "Professional Services", "Accounting & Audit"),
    "advertising":        ("Expense", "Operating", "Marketing", "Advertising"),
    "hotel":              ("Expense", "Operating", "Travel", "Accommodation"),
    "air_travel":         ("Expense", "Operating", "Travel", "Air Travel"),
    "freight":            ("Expense", "Cost of Sales", "Logistics", "Freight"),
    "raw_materials":      ("Expense", "Cost of Sales", "Materials", "Raw Materials"),
    "packaging":          ("Expense", "Cost of Sales", "Materials", "Packaging"),
    "office_supplies":    ("Expense", "Operating", "Office", "Office Supplies"),
}


def category_path(key: str) -> str:
    return " > ".join(TAXONOMY[key])


@dataclass(frozen=True)
class ItemSpec:
    category: str
    descriptions: tuple[str, ...]
    hsn_codes: tuple[str, ...]
    rate_before: Decimal
    rate_after: Decimal
    unit_price: tuple[int, int]
    quantity: tuple[int, int]
    tds_rate: Decimal = Decimal("0")

    @property
    def hsn_sac(self) -> str:
        return self.hsn_codes[0]


def _r(before: str, after: str | None = None) -> tuple[Decimal, Decimal]:
    return Decimal(before), Decimal(after if after is not None else before)


ITEMS: tuple[ItemSpec, ...] = (
    ItemSpec("cloud_services",
             ("Cloud hosting charges", "Managed database instance", "Object storage usage",
              "Compute instance charges", "Hosting services - monthly", "Cloud infra charges",
              "Virtual server rental", "Backup storage services"),
             ("998315", "998313", "998316"), *_r("18"), (4_000, 90_000), (1, 1)),
    ItemSpec("software_licences",
             ("Annual software subscription", "User licence renewal", "Accounting software licence",
              "SaaS subscription charges", "Software maintenance renewal", "Licence fees - annual",
              "Per seat licence charges", "Productivity suite subscription"),
             ("997331", "998434"), *_r("18"), (2_500, 60_000), (1, 25)),
    ItemSpec("it_consulting",
             ("IT support retainer", "Website development milestone", "Network setup services",
              "Technical consultancy charges", "System integration services", "Application support services",
              "Server migration services", "IT services - monthly"),
             ("998313", "998314"), *_r("18"), (15_000, 250_000), (1, 1), Decimal("2")),
    ItemSpec("it_hardware",
             ("Laptop 14in i5 16GB", "Wireless keyboard and mouse", "27in LED monitor",
              "Desktop workstation", "External SSD 1TB", "Network switch 24 port",
              "Docking station", "Webcam full HD"),
             ("8471", "84713010", "8528"), *_r("18"), (1_200, 85_000), (1, 10)),
    ItemSpec("furniture",
             ("Ergonomic office chair", "Workstation desk", "Filing cabinet steel",
              "Conference table", "Storage rack", "Visitor chair set",
              "Modular partition panel", "Reception counter"),
             ("9401", "9403"), *_r("18"), (3_500, 28_000), (1, 12)),
    ItemSpec("rent",
             ("Office rent for the month", "Warehouse lease rental", "Commercial premises rent",
              "Monthly rental charges", "Godown rent", "Office space lease - monthly",
              "Rent as per agreement", "Premises rental charges"),
             ("997212", "997213"), *_r("18"), (35_000, 400_000), (1, 1), Decimal("10")),
    ItemSpec("telecom",
             ("Internet leased line charges", "Broadband plan monthly", "Enterprise mobile postpaid",
              "Fibre connectivity charges", "Data circuit rental", "Telecom services - monthly",
              "Bandwidth charges", "SIP trunk charges"),
             ("998422", "998424"), *_r("18"), (1_500, 45_000), (1, 1)),
    ItemSpec("repairs",
             ("AC servicing and repair", "Electrical maintenance work", "Cement OPC 53 grade bag",
              "Plumbing repair charges", "Annual maintenance contract", "Civil repair work",
              "Equipment servicing charges", "Painting and touch up work"),
             ("998719", "995461", "2523"), *_r("18"), (1_200, 60_000), (1, 5), Decimal("2")),
    ItemSpec("legal",
             ("Legal advisory fees", "Contract drafting and review", "Trademark filing assistance",
              "Retainer fees - legal", "Litigation support charges", "Compliance advisory services",
              "Agreement vetting charges", "Legal opinion fees"),
             ("998211", "998212"), *_r("18"), (10_000, 180_000), (1, 1), Decimal("10")),
    ItemSpec("accounting",
             ("Statutory audit fees", "Monthly bookkeeping services", "GST return filing fees",
              "Tax advisory charges", "Accounting services - monthly", "TDS return filing charges",
              "Internal audit fees", "Financial statement preparation"),
             ("998221", "998222"), *_r("18"), (5_000, 150_000), (1, 1), Decimal("10")),
    ItemSpec("advertising",
             ("Digital ad campaign management", "Print advertisement insertion", "Hoarding display rental",
              "Social media promotion charges", "Search ad spends", "Creative design services",
              "Radio spot charges", "Brand promotion services"),
             ("998361", "998365"), *_r("18"), (8_000, 300_000), (1, 1), Decimal("2")),
    ItemSpec("hotel",
             ("Room tariff - deluxe", "Accommodation charges 2 nights", "Business hotel stay",
              "Lodging charges", "Guest room charges", "Stay charges - executive room",
              "Hotel booking charges", "Room rent and taxes"),
             ("996311", "996312"), *_r("12", "5"), (2_800, 7_000), (1, 4)),
    ItemSpec("air_travel",
             ("Air ticket economy class", "Domestic flight booking", "Return airfare",
              "Flight charges - sector fare", "Air travel booking charges", "Ticket charges economy",
              "Airfare and convenience fee", "Sector fare charges"),
             ("996425", "996422"), *_r("5"), (3_500, 18_000), (1, 3)),
    ItemSpec("freight",
             ("Road freight charges", "Goods transport by truck", "Local delivery charges",
              "Transportation charges", "Courier and delivery charges", "Cartage and handling charges",
              "Consignment freight charges", "Logistics services charges"),
             ("996511", "996519"), *_r("5"), (2_000, 75_000), (1, 1), Decimal("2")),
    ItemSpec("raw_materials",
             ("HR steel sheet coil", "Plastic granules HDPE", "Cotton yarn 40s count",
              "MS angle sections", "Polymer resin grade A", "Aluminium sheet",
              "Copper wire rod", "Fabric roll grey"),
             ("7208", "3901", "5205"), *_r("18"), (60, 1_800), (50, 2_000)),
    ItemSpec("packaging",
             ("Corrugated cartons 5 ply", "Stretch wrap film roll", "Bubble wrap roll",
              "Packaging boxes printed", "Shrink film roll", "Strapping roll",
              "Carton boxes plain", "Packing material assorted"),
             ("4819", "3923"), *_r("12", "5"), (15, 900), (50, 3_000)),
    ItemSpec("office_supplies",
             ("A4 copier paper ream", "Printer toner cartridge", "Stationery kit assorted",
              "Files and folders", "Whiteboard markers pack", "Notebooks and registers",
              "Pantry consumables", "Housekeeping supplies"),
             ("4802", "4820", "8443"), *_r("12", "5"), (180, 6_500), (1, 60)),
)

ITEMS_BY_CATEGORY: dict[str, tuple[ItemSpec, ...]] = {
    key: tuple(spec for spec in ITEMS if spec.category == key) for key in TAXONOMY
}
ALL_HSN_CODES: tuple[str, ...] = tuple({code for spec in ITEMS for code in spec.hsn_codes})

# How suppliers actually write a line: abbreviations, references and periods attached to the phrase.
DESCRIPTION_PREFIXES = ("", "", "", "", "Being ", "Charges for ", "Supply of ")
DESCRIPTION_SUFFIXES = ("", "", "", "", " - {month}", " {month}-{yy}", " as per PO {po}",
                        " ref {po}", " (Qty as per order)", " - {month} {yy}")
ABBREVIATIONS = {"charges": "chgs", "services": "svcs", "maintenance": "maint",
                 "monthly": "mthly", "annual": "annl", "advertisement": "advt"}

# Informative name cores leak category signal on purpose; generic names force models to read line items.
INFORMATIVE_CORES: dict[str, tuple[str, ...]] = {
    "cloud_services": ("Nimbus Cloud", "Stratus Hosting", "CloudNest"),
    "software_licences": ("Codeleaf Software", "Ledgerly Systems", "Bytecraft"),
    "it_consulting": ("Techvista IT", "Netcore Solutions", "Sysbridge"),
    "it_hardware": ("Digital Hub Computers", "Micro Point", "Laptop World"),
    "furniture": ("Comfort Seating", "Woodline Interiors", "Office Craft"),
    "rent": ("Skyline Realty", "Prime Estates", "Metro Properties"),
    "telecom": ("Linkup Broadband", "Fibernet", "Airwave Telecom"),
    "repairs": ("Coolcare Services", "Powerfix Electricals", "BuildRight"),
    "legal": ("Mehta Legal", "Iyer & Rao Law", "Lexpoint"),
    "accounting": ("Kapoor & Associates", "Numbers Tax Consultants", "Ledger Point"),
    "advertising": ("Brandwave Media", "Adstreet", "Pixel Promotions"),
    "hotel": ("Grand Residency", "Comfort Inn Suites", "Royal Stay"),
    "air_travel": ("Skyways Travels", "Jetset Tours", "Aerolink Travels"),
    "freight": ("Speedline Logistics", "Highway Carriers", "Swift Transport"),
    "raw_materials": ("Tata Metal Traders", "Polymer House", "Spintex Yarns"),
    "packaging": ("Boxpack Industries", "Wrapwell Packaging", "Cartonix"),
    "office_supplies": ("Paperworld Stationers", "Office Mart", "Inkjet Point"),
}
GENERIC_PREFIXES = ("Shree", "Sai", "Om", "Balaji", "Krishna", "Ganesh", "Jai Mata Di", "Sri Lakshmi", "Apex", "Vertex")
GENERIC_CORES = ("Enterprises", "Traders", "Industries", "Agencies", "Corporation", "Sales", "Distributors")
GENERIC_MIDDLES = ("", "", "", "Global", "India", "Mega", "New", "Royal", "Star", "United", "Supreme", "Techno")
SURNAMES = ("Sharma", "Patel", "Reddy", "Nair", "Gupta", "Iyer", "Singh", "Das", "Joshi", "Menon", "Khan", "Rao")
LEGAL_SUFFIX = {"proprietorship": "", "firm": " & Co.", "llp": " LLP", "company": " Pvt Ltd"}
STREETS = ("MG Road", "Station Road", "Industrial Area Phase II", "Ring Road", "Link Road", "Nehru Nagar", "Sector 18")
