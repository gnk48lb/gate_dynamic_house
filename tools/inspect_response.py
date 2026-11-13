"""Inspect the raw Magpie payload shape for a benchmark-like prompt."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import BASE_SYSTEM  # noqa: E402
from env import DynamicHouseEnv  # noqa: E402

env = DynamicHouseEnv(4, 0.4)
prompt = json.dumps({"task_state": env.public_state(), "instruction": "Choose exactly one next action."},
                    ensure_ascii=False)
model = sys.argv[1] if len(sys.argv) > 1 else "codearts/deepseek-v4.1-flash-1017"
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": prompt}]}
request = urllib.request.Request("http://127.0.0.1:3425/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Authorization": "Bearer local", "Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=120) as response:
    payload = json.loads(response.read().decode("utf-8"))
print("top-level keys:", sorted(payload))
print("usage:", payload.get("usage"))
choice = (payload.get("choices") or [{}])[0]
print("choice keys:", sorted(choice))
message = choice.get("message", {})
print("message keys:", sorted(message))
for key, value in message.items():
    if isinstance(value, str):
        print(f"  {key}: str len={len(value)} -> {value[:200]!r}")
    elif isinstance(value, list):
        print(f"  {key}: list len={len(value)}")
        for part in value[:4]:
            print("     ", json.dumps(part, ensure_ascii=False)[:300])
    else:
        print(f"  {key}: {value!r}")
print("finish_reason:", choice.get("finish_reason"))
