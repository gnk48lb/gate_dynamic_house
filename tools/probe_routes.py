"""Probe candidate fixed routes with repeated concurrent action requests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import MagpieClient, parse_action  # noqa: E402

CANDIDATES = {
    "deepseek@codearts": "codearts/deepseek-v4.1-flash-1017",
    "glm@codearts": "codearts/glm-5.3-flash",
    "minimax@global": "minimax-code-global/MiniMax-M3.1-Flash",
}
SYSTEM = ('Return exactly one JSON object with your next house action. Allowed: '
          '{"action":"observe"} or {"action":"move","room":"Kitchen"}. No prose.')
USER = "You are in LivingRoom. Neighbors: Kitchen, Bedroom, Study. Reply with one action."


def probe(label, model):
    started = time.monotonic()
    try:
        text, meta = MagpieClient(timeout=90, max_retries=2).complete(model, SYSTEM, USER)
        parse_action(text)
        return {"label": label, "ok": True, "latency_s": round(time.monotonic() - started, 2),
                "tokens": meta.get("total_tokens")}
    except Exception as exc:
        return {"label": label, "ok": False, "latency_s": round(time.monotonic() - started, 2),
                "error": f"{type(exc).__name__}: {exc}"[:240]}


jobs = [(label, model) for label, model in CANDIDATES.items() for _ in range(10)]
results = []
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = [pool.submit(probe, *job) for job in jobs]
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
print("--- summary ---")
for label in CANDIDATES:
    rows = [row for row in results if row["label"] == label]
    ok = [row for row in rows if row["ok"]]
    latencies = [row["latency_s"] for row in ok]
    print(json.dumps({"label": label, "success": f"{len(ok)}/{len(rows)}",
                      "median_ok_latency_s": round(statistics.median(latencies), 2) if latencies else None}))
