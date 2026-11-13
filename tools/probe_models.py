"""One-call liveness and JSON-protocol probe for the fixed benchmark models."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import MagpieClient, parse_action  # noqa: E402
from run_benchmark import resolve_models  # noqa: E402


def probe(name, model):
    text, meta = MagpieClient(timeout=120, max_retries=1).complete(
        model,
        "Return exactly one JSON object with your next house action. Allowed: "
        '{"action":"observe"} or {"action":"move","room":"Kitchen"}. No prose.',
        'You are in LivingRoom. Neighbors: Kitchen, Bedroom, Study. Reply with one action.',
    )
    action = parse_action(text)
    return name, model, action, meta


with ThreadPoolExecutor(max_workers=3) as pool:
    for name, model, action, meta in pool.map(lambda item: probe(*item),
                                              resolve_models("http://127.0.0.1:3425", "local", "deepseek,kimi,glm").items()):
        print(json.dumps({"preset": name, "model": model, "action": action,
                          "tokens": meta["total_tokens"], "latency_ms": meta["latency_ms"]},
                         ensure_ascii=False))
