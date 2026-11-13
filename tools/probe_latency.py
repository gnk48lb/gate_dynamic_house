"""Compare candidate routes: latency, throughput and failure rate."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import MagpieClient  # noqa: E402

SYSTEM = ('Return exactly one flat JSON object with your next house action. Allowed: '
          '{"action":"observe"} or {"action":"move","room":"Kitchen"}. No prose.')
USER = "You are in LivingRoom. Neighbors: Kitchen, Bedroom, Study. Reply with one action."

CANDIDATES = {
    "autoclaw/zaicoding": "autoclaw-cn/zaicoding_glm-5.3",
    "autoclaw/zai_flash": "autoclaw-cn/zai_glm-5.3-flash",
    "codearts/deepseek": "codearts/deepseek-v4.1-flash-1017",
    "codearts/glm_flash": "codearts/glm-5.3-flash",
    "minimax/global": "minimax-code-global/MiniMax-M3.1-Flash",
}


def one(client, model):
    started = time.monotonic()
    try:
        text, meta = client.complete(model, SYSTEM, USER)
        ok = bool(text.strip())
        return {"ok": ok, "ms": round((time.monotonic() - started) * 1000),
                "tokens": meta.get("total_tokens"),
                "text": text[:60]}
    except Exception as exc:
        return {"ok": False, "ms": round((time.monotonic() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"[:140]}


mode = sys.argv[1] if len(sys.argv) > 1 else "single"
if mode == "single":
    # One request at a time: measures the route's intrinsic latency.
    for label, model in CANDIDATES.items():
        client = MagpieClient(timeout=180, max_retries=1, max_tokens=2000)
        results = [one(client, model) for _ in range(3)]
        ok = [r for r in results if r["ok"]]
        print(json.dumps({"label": label, "model": model, "ok": f"{len(ok)}/{len(results)}",
                          "median_ms": int(statistics.median(r["ms"] for r in ok)) if ok else None,
                          "max_ms": max((r["ms"] for r in ok), default=None),
                          "error": next((r["error"] for r in results if not r["ok"]), None),
                          "sample": ok[0]["text"] if ok else None}), flush=True)
else:
    # Sustained load: 12 requests at a given concurrency.
    label, model, concurrency = sys.argv[2], sys.argv[3], int(sys.argv[4])
    client = MagpieClient(timeout=240, max_retries=1, max_tokens=2000)
    results = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in as_completed([pool.submit(one, client, model) for _ in range(12)]):
            results.append(future.result())
    span = time.monotonic() - started
    ok = [r for r in results if r["ok"]]
    print(json.dumps({"label": label, "concurrency": concurrency, "ok": f"{len(ok)}/{len(results)}",
                      "failure_rate": round(1 - len(ok) / len(results), 3),
                      "throughput_rpm": round(len(ok) / span * 60, 1),
                      "median_ms": int(statistics.median(r["ms"] for r in ok)) if ok else None,
                      "max_ms": max((r["ms"] for r in ok), default=None),
                      "errors": [r["error"] for r in results if not r["ok"]][:2]}), flush=True)
