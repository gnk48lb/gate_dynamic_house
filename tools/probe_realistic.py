"""Measure a route's throughput with a REALISTIC prompt (full ledger state).

Short probe prompts understate latency: the benchmark sends a ~1-2k token
context every step, so the upstream sees a much bigger payload. This loads the
actual agent prompt to find the concurrency that actually scales.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import BASE_SYSTEM, MagpieClient  # noqa: E402
from env import DynamicHouseEnv  # noqa: E402

MODEL = sys.argv[1]
CONCURRENCY = int(sys.argv[2])
REQUESTS = int(sys.argv[3]) if len(sys.argv) > 3 else 12

env = DynamicHouseEnv(4, 0.4)
prompt = json.dumps({"task_state": env.public_state(),
                      "room_graph": {"LivingRoom": ["Kitchen"]},
                      "versioned_state_ledger": {item: {"loc": "Kitchen", "memory_age": 9,
                                                        "risk_level": "low_risk", "superseded_count": 1}
                                                for item in list(env.locations)},
                      "recent_history": []}, ensure_ascii=False)


def one(client):
    started = time.monotonic()
    try:
        text, meta = client.complete(MODEL, BASE_SYSTEM, prompt)
        return {"ok": bool(text.strip()), "ms": round((time.monotonic() - started) * 1000),
                "in_tokens": meta.get("input_tokens"), "out_tokens": meta.get("output_tokens")}
    except Exception as exc:
        return {"ok": False, "ms": round((time.monotonic() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"[:120]}


client = MagpieClient(timeout=180, max_retries=1, max_tokens=2000)
results = []
started = time.monotonic()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    for future in as_completed([pool.submit(one, client) for _ in range(REQUESTS)]):
        results.append(future.result())
span = time.monotonic() - started
ok = [r for r in results if r["ok"]]
print(json.dumps({
    "model": MODEL, "concurrency": CONCURRENCY, "ok": f"{len(ok)}/{len(results)}",
    "throughput_rpm": round(len(ok) / span * 60, 1),
    "median_ms": int(statistics.median(r["ms"] for r in ok)) if ok else None,
    "max_ms": max((r["ms"] for r in ok), default=None),
    "median_in_tokens": int(statistics.median(r["in_tokens"] for r in ok)) if ok else None,
    "errors": [r["error"] for r in results if not r["ok"]][:2],
}), flush=True)
