"""Create a tiny sample delay-report PDF for portal demos."""
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

# Minimal PDF via report-less approach: write plain text file fallback if needed
OUT = Path(__file__).resolve().parents[1] / "data" / "demo" / "sample_delay_report.pdf"

try:
    from pypdf.generic import create_string_object  # type: ignore
except Exception:
    create_string_object = str  # type: ignore

# Use a very small handcrafted PDF
content = b"""BT /F1 12 Tf 50 750 Td (Construction Delay Risk Report) Tj 0 -20 Td (Project: Demo Capital Facility) Tj 0 -20 Td (1. Procurement delayed due to bid protests and contract awards.) Tj 0 -20 Td (2. Scope design changes required structural redesign.) Tj 0 -20 Td (3. External utility dependency blocked excavation permits.) Tj 0 -20 Td (4. Workforce resource shortages on critical path crews.) Tj ET"""

# Simpler: write text sidecar and a note; for real PDF use fpdf if available
text_path = OUT.with_suffix(".txt")
text_path.write_text(
    "Construction Delay Risk Report\n"
    "Project: Demo Capital Facility\n"
    "1. Procurement delayed due to bid protests and contract awards.\n"
    "2. Scope design changes required structural redesign.\n"
    "3. External utility dependency blocked excavation permits.\n"
    "4. Workforce resource shortages on critical path crews.\n",
    encoding="utf-8",
)

# Build minimal PDF
pdf = (
    b"%PDF-1.4\n"
    b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
    b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
)
stream = b"BT /F1 11 Tf 50 720 Td (Construction Delay Risk Report) Tj T* (Procurement delayed due to bid protests.) Tj T* (Scope design changes required redesign.) Tj T* (External utility dependency blocked work.) Tj T* (Workforce resource shortages on site.) Tj ET"
pdf += f"4 0 obj<< /Length {len(stream)} >>stream\n".encode() + stream + b"\nendstream\nendobj\n"
pdf += b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
xref_pos = len(pdf)
pdf += b"xref\n0 6\n0000000000 65535 f \n"
# crude xref omitted precision ok for demo extractors that tolerate
pdf += b"trailer<< /Size 6 /Root 1 0 R >>\nstartxref\n"
pdf += str(xref_pos).encode() + b"\n%%EOF\n"
OUT.write_bytes(pdf)
print("wrote", OUT)
