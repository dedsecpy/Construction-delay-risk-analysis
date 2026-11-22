import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

# multipart form without extra deps
boundary = "----SentinelBoundary7MA4YWxkTrZu0gW"
notes = "Procurement delayed due to contract disputes. Scope design changes required redesign."
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="pid"\r\n\r\n137\r\n'
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="budget"\r\n\r\n2\r\n'
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="policy"\r\n\r\ndisagreement_voi\r\n'
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="notes"\r\n\r\n{notes}\r\n'
    f"--{boundary}--\r\n"
).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/analyze",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
r = json.load(urllib.request.urlopen(req))
print("ok", r.get("ok"), "p", round(r["final_p_severe"], 3), "reasons", len(r.get("mapped_reasons", [])), "alert", r["alerted"])

opts = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/projects/options?split=test&limit=3"))
print("options", opts["total"], opts["items"][0]["name"][:40])
