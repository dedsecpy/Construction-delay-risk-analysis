import json
import urllib.request

print(json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/health")))
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/inspect/run",
    data=json.dumps({"pid": "103", "policy": "disagreement_voi", "budget": 2}).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req))
print(r["pid"], len(r["steps"]), round(r["final_p_severe"], 3), r["alerted"], r["correct"])
