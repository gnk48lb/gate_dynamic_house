"""Find the fastest safe concurrency for the CodeArts account.

CodeArts documents a hard 3-concurrent-session cap; Magpie reuses one session
id per account, so excess concurrency becomes soft queueing that trips our
wall-clock deadline. This measures failure rate and throughput per setting.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import MagpieClient  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "codearts/deepseek-v4.1-flash-1017"
SYSTEM = ('Return exactly one flat JSON object with your next house action. Allowed: '
          '{"action":"observe"} or {"action":"move","room":"Kitchen"}. No prose.')
USER = "You are in LivingRoom. Neighbors: Kitchen, Bedroom, Study. Reply with one action."


def one(client, index):
    started = time.monotonic()
    try:
        text, meta = client.complete(MODEL, SYSTEM, USER)
        return {"ok": True, "ms": round((time.monotonic() - started) * 1000),
                "tokens": meta.get("total_tokens")}
    except Exception as exc:
        return {"ok": False, "ms": round((time.monotonic() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"[:120]}


for concurrency in (3, 4, 6, 8):
    client = MagpieClient(timeout=180, max_retries=1, max_tokens=2000)
    results = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in as_completed([pool.submit(one, client, i) for i in range(18)]):
            results.append(future.result())
    span = time.monotonic() - started
    ok = [r for r in results if r["ok"]]
    latencies = [r["ms"] for r in ok]
    print(json.dumps({
        "concurrency": concurrency, "n": len(results), "ok": len(ok),
        "failure_rate": round(1 - len(ok) / len(results), 3),
        "throughput_rpm": round(len(ok) / span * 60, 1),
        "median_ms": int(statistics.median(latencies)) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
    }), flush=True)
