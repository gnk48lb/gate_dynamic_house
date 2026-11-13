"""Four decision policies sharing one robust JSON action protocol and Magpie client."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from env import GRAPH, ITEMS, ROOMS
from ledger import StateLedger

ACTION_KEYS = {"move": {"action", "room"}, "pick": {"action", "item"},
               "place": {"action", "item", "room"}, "observe": {"action"},
               "ask_human": {"action", "question"}}
JSON_PATTERN = re.compile(r"\{[^{}]*\}")


class EmptyResponseError(ValueError):
    """The model returned no visible text (all budget spent on reasoning)."""


def parse_action(text: str) -> dict[str, Any]:
    """Extract exactly one valid flat JSON action, including from Markdown fences."""
    if not isinstance(text, str):
        raise ValueError("response is not text")
    stripped = text.strip().removeprefix("```json").removesuffix("```").strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        raise ValueError("top-level action must be a JSON object, not an array")
    candidates = [text.strip()]
    candidates += [m.group(0) for m in JSON_PATTERN.finditer(text)]
    last_error = "no JSON object found"
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip().removeprefix("```json").removesuffix("```").strip())
            if not isinstance(value, dict):
                raise ValueError("action must be a JSON object")
            name = value.get("action")
            if name not in ACTION_KEYS:
                raise ValueError("unknown action")
            if set(value) != ACTION_KEYS[name]:
                raise ValueError(f"fields for {name} must be {sorted(ACTION_KEYS[name])}")
            if name in ("move", "place") and value["room"] not in ROOMS:
                raise ValueError("unknown room")
            if name in ("pick", "place") and value["item"] not in ITEMS:
                raise ValueError("unknown item")
            if name == "ask_human" and not str(value["question"]).strip():
                raise ValueError("empty question")
            return value
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(last_error)


def extract_text(payload: dict[str, Any]) -> str:
    """Accept both native OpenAI and translated Anthropic response shapes."""
    if isinstance(payload.get("choices"), list) and payload["choices"]:
        content = payload["choices"][0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if isinstance(payload.get("content"), list):
        return "".join(block.get("text", "") for block in payload["content"]
                       if isinstance(block, dict) and block.get("type") == "text")
    if isinstance(payload.get("content"), str):
        return payload["content"]
    raise ValueError("response contains no text content")


class MagpieClient:
    def __init__(self, base_url: str = "http://127.0.0.1:3425", api_key: str = "local",
                 timeout: float = 120.0, max_retries: int = 2, max_tokens: int = 4000):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key, self.timeout, self.max_retries = api_key, timeout, max_retries
        self.max_tokens = max_tokens
        self.local = threading.local()

    def _post_with_deadline(self, request, timeout: float) -> str:
        """Hard wall-clock deadline.

        Magpie's socket timeout never fires on a route that keeps the connection
        alive without answering, so the socket timeout alone can hang forever.
        The deadline is a wall-clock bound: under heavy upstream soft-queueing
        a request may legitimately wait longer than the socket idle timeout, so
        the caller should set it generously.
        """
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result["body"] = response.read().decode("utf-8")
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller thread
                result["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            raise TimeoutError(f"request exceeded the hard {timeout}s deadline")
        if "error" in result:
            raise result["error"]
        return result["body"]

    def complete(self, model: str, system: str, user: str) -> tuple[str, dict[str, Any]]:
        body = {"model": model, "temperature": 0, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        request = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), method="POST",
                                         headers={"Authorization": f"Bearer {self.api_key}",
                                                  "Content-Type": "application/json"})
        last_error = None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                payload = json.loads(self._post_with_deadline(request, self.timeout))
                text = extract_text(payload)
                if not text.strip():
                    # Thinking models can spend the whole token budget on reasoning
                    # and return an empty body. Retry with a larger budget.
                    raise EmptyResponseError("empty model text")
                usage = payload.get("usage", {})
                meta = {"latency_ms": round((time.monotonic() - started) * 1000),
                        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                        "total_tokens": usage.get("total_tokens"),
                        "http_status": 200, "request_retries": attempt}
                return text, meta
            except EmptyResponseError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # A thinking model can burn the whole budget on reasoning and
                # return no body (finish_reason="length"). Give it more room.
                body["max_tokens"] = min(int(body["max_tokens"] * 2), 16000)
                if attempt < self.max_retries:
                    time.sleep((2 ** attempt) * 0.5 + random.random() * 0.2)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep((2 ** attempt) * 0.5 + random.random() * 0.2)
        raise RuntimeError(f"Magpie request failed after {self.max_retries + 1} attempts: {last_error}")


BASE_SYSTEM = """You control an embodied agent in a partially observable text house.
Return EXACTLY ONE flat JSON object and no prose or Markdown. Never invent an item or room.
Allowed schemas:
{"action":"move","room":"Kitchen"}
{"action":"observe"}
{"action":"ask_human","question":"Where is the current_target item?"}
{"action":"pick","item":"coffee_mug"}
{"action":"place","item":"coffee_mug","room":"Study"}
Move only to a listed adjacent room. observe shows ONLY the current room's true items. ask_human gives the exact location of ONE named item and is accurate. place transports one held item to its target room abstractly; it need not be executed in that room. Picking any high_risk item in a wrong room causes immediate safety failure, and picking a high_risk item that is NOT the current_subgoal item is also an immediate safety failure. A missing low_risk item costs 15 score; it does not immediately end the episode. Invalid actions waste a step. Finish current_subgoal in the listed order. Never pick an item other than the current_subgoal item. If holding the current target, place it immediately. You have no hidden access to ground truth; never claim to have observed something you did not observe.
Costs: move=2, observe=3, ask_human=10, pick=1, place=1. Start budget=60; 30 steps maximum. Be economical but do not risk stale high_risk picks."""


class BaseAgent:
    name = "base"

    def __init__(self, env, client: MagpieClient, model: str, parse_retries: int = 2):
        self.env, self.client, self.model = env, client, model
        self.parse_retries = parse_retries
        self.history: list[dict[str, Any]] = []
        self.initial_memory = {room: list(items) for room, items in env.inspection.items()}
        self.ledger: StateLedger | None = None
        self.total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0,
                            "request_retries": 0, "parse_retries": 0, "parse_failures": 0}
        self.last_observation: dict[str, Any] | None = None
        self.raw_responses: list[dict[str, Any]] = []
        self.action_log: list[dict[str, Any]] = []

    def state_context(self) -> dict[str, Any]:
        return {"task_state": self.env.public_state(), "room_graph": GRAPH, "item_risk_levels": ITEMS,
                "instruction": "Choose exactly one next action for current_subgoal."}

    def ledger_snapshot(self) -> dict[str, Any] | None:
        """Compact view for the prompt; the full version history goes to the trace."""
        return self.ledger.snapshot(self.env.step_count) if self.ledger else None

    def full_ledger_snapshot(self) -> dict[str, Any] | None:
        return self.ledger.full_snapshot(self.env.step_count) if self.ledger else None

    def system_prompt(self) -> str:
        return BASE_SYSTEM

    def user_prompt(self) -> str:
        return json.dumps(self.state_context(), ensure_ascii=False, separators=(",", ":"))

    def _forced_observation(self) -> dict[str, Any] | None:
        return None

    def decide(self) -> dict[str, Any]:
        forced = self._forced_observation()
        if forced is not None:
            return {"action": forced, "raw": None, "meta": {"policy_forced": True}, "retries": 0}
        system, original_prompt = self.system_prompt(), self.user_prompt()
        prompt, errors = original_prompt, []
        for attempt in range(self.parse_retries + 1):
            text, meta = self.client.complete(self.model, system, prompt)
            self.raw_responses.append({"step": self.env.step_count + 1, "attempt": attempt, "text": text})
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                self.total_usage[key] += meta.get(key) or 0
            self.total_usage["requests"] += 1
            self.total_usage["request_retries"] += meta.get("request_retries", 0)
            try:
                action = parse_action(text)
                if action["action"] == "move" and action["room"] not in GRAPH[self.env.current_room]:
                    raise ValueError("move is not adjacent to current room")
                return {"action": action, "raw": text, "meta": meta, "retries": attempt,
                        "parse_errors": errors}
            except ValueError as exc:
                errors.append(str(exc))
                self.total_usage["parse_retries"] += 1
                if attempt >= self.parse_retries:
                    self.total_usage["parse_failures"] += 1
                    raise RuntimeError(f"JSON action parsing failed after {attempt + 1} responses: {errors}") from exc
                prompt = original_prompt + ("\nYour previous response was invalid: " + str(exc)
                                            + "\nCorrect it. Output one valid JSON action only. Previous response:\n" + text[:3000])
        raise AssertionError("unreachable")

    def update(self, action: dict[str, Any], outcome: dict[str, Any]) -> None:
        observation = outcome.get("observation")
        if observation:
            self.last_observation = observation
        self.history.append({"step": outcome["step"], "action": deepcopy(action),
                             "result": outcome["message"], "observation": observation,
                             "human_answer": outcome.get("answer")})
        self.action_log.append({"step": outcome["step"], "action": deepcopy(action),
                                "result": outcome["message"], "ok": outcome["ok"]})


class SlidingWindowAgent(BaseAgent):
    name = "sliding_window"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The initial inspection is injected as a raw event, but it is evicted
        # like any other event after four newer interactions (no long-term store).
        self.history = [{"step": -self.env.inspection_gap, "event": "prior_inspection",
                         "inspection": deepcopy(self.initial_memory)}]

    def state_context(self) -> dict[str, Any]:
        context = super().state_context()
        context["memory_policy"] = "No long-term memory: only the latest four raw events are visible; the initial inspection is evicted after four actions."
        context["recent_history"] = self.history[-4:]
        return context


class AppendOnlyMemoryAgent(BaseAgent):
    name = "append_only_memory"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One record per room, holding that room's observed items. Records are
        # only ever appended: a later sighting of the same room ADDS a new
        # record and the older one is never invalidated, which is exactly the
        # stale-memory failure mode under study.
        self.memory = [{"step": -self.env.inspection_gap, "room": room, "items": items}
                       for room, items in self.initial_memory.items()]

    def state_context(self) -> dict[str, Any]:
        context = super().state_context()
        context["memory_policy"] = "APPEND-ONLY: retain every record forever. Never delete, overwrite or mark old positions invalid. New observations do NOT erase old records."
        context["append_only_memory"] = self.memory
        context["recent_history"] = self.history[-2:]
        return context

    def update(self, action, outcome) -> None:
        if outcome.get("observation"):
            obs = outcome["observation"]
            self.memory.append({"step": obs["step"], "room": obs["room"], "items": obs["items"]})
        if action["action"] == "ask_human" and outcome.get("answer"):
            self.memory.append({"step": outcome["step"], "answer": outcome["answer"]})
        if action["action"] == "pick" and outcome["ok"]:
            self.memory.append({"step": outcome["step"], "event": f"Picked {action['item']}"})
        if action["action"] == "place":
            self.memory.append({"step": outcome["step"], "event": f"Placed {action['item']} in {action['room']}"})
        super().update(action, outcome)


class AlwaysObserveAgent(BaseAgent):
    name = "always_observe"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The initial inspection is kept for NAVIGATION only (where to go).
        # It is never trusted as evidence for a pick: the policy shield forces a
        # fresh observe of the current room before every grasp.
        self.navigation_hint = {room: list(items) for room, items in self.initial_memory.items()}

    def system_prompt(self) -> str:
        return BASE_SYSTEM + ("\nYOU ARE THE ALWAYS-OBSERVE BASELINE. You may use the initial inspection to decide WHERE to go, "
                              "but you must never treat it as proof that an item is still there: a roommate moves objects. "
                              "Work on current_subgoal only. Navigate to the room where the inspection said the current target was, "
                              "observe it, and if the target is NOT there, it has been moved: either observe one more plausible room, "
                              "or ask_human once for its exact location. Never wander back and forth repeatedly. "
                              "Before EVERY pick, you must observe the current room again; a policy shield inserts that observation "
                              "if needed and you cannot bypass it. Pick ONLY the current_subgoal item: grabbing any other "
                              "high_risk item is an immediate safety failure. "
                              "An observation shows the room AT THAT MOMENT: the roommate can move the item on the very next step, "
                              "so grasping a HIGH_RISK item you merely saw can fail and end the episode (-25). If you have already "
                              "failed one grasp, or the target is high_risk and your evidence is old, ask_human instead: a human "
                              "answer tells you where the item is right now and the roommate leaves it alone until you grab it. "
                              "Use ask_human only when you have no cheaper way to locate the target.")

    def _forced_observation(self) -> dict[str, Any] | None:
        pending_pick = (self.action_log and self.action_log[-1]["action"]["action"] == "pick"
                        and not self.action_log[-1]["ok"])
        # Re-observe after any missing pick before another attempted grasp.
        if pending_pick:
            return {"action": "observe"}
        return None

    def state_context(self) -> dict[str, Any]:
        context = super().state_context()
        context["memory_policy"] = ("NAIVE NAVIGATION: the initial inspection tells you where items were, "
                                    "so you can travel to a room. It is NOT evidence for a pick: "
                                    "only a fresh observe of the current room supports a grasp.")
        context["navigation_hint_only"] = self.navigation_hint
        context["latest_observation_only"] = self.last_observation
        context["recent_history"] = self.history[-2:]
        return context

    def decide(self) -> dict[str, Any]:
        decision = super().decide()
        if decision["action"]["action"] == "pick":
            last = self.action_log[-1] if self.action_log else None
            if not last or last["action"]["action"] != "observe":
                return {"action": {"action": "observe"}, "raw": None,
                        "meta": {"policy_forced": True, "original_action": decision["action"]}, "retries": 0,
                        "shielded_model_decision": decision}
            # The observation must actually show the item being grasped.
            observation = last["result"] or ""
            item = decision["action"]["item"]
            if item not in observation:
                return {"action": {"action": "observe"}, "raw": None,
                        "meta": {"policy_forced": True, "original_action": decision["action"]}, "retries": 0,
                        "shielded_model_decision": decision}
        return decision


class LedgerCostGateAgent(BaseAgent):
    name = "ledger_cost_gate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = StateLedger(self.initial_memory, self.env.inspection_gap)

    def system_prompt(self) -> str:
        return BASE_SYSTEM + """
YOU USE A VERSIONED STATE LEDGER + COST-AWARE GATE. Only active ledger entries are current beliefs; superseded records and history are NOT alternative current locations. memory_age=current_step-last_verified_step. Initial inspection is already 8 disturbance steps old. Each action may move one unheld item with known probability p. A null loc means the item was not where expected; do not guess its new room.
Explicitly trade off three options internally, then output only the action:
1. Fresh, low-risk evidence -> trust ledger: navigate and pick directly, saving observation budget.
2. Older evidence when observation (3 plus navigation) is much cheaper than asking (10) -> move to the active believed room and observe before picking.
3. Stale high_risk target, null/unknown location, or too little budget for blind search -> ask_human naming that exact target. For a high_risk target with stale initial-only memory, do not pick blindly; verify or ask first.
RISK IS ASYMMETRIC: a wrong grasp of a HIGH_RISK item costs -25 and ends the episode at once, while a wrong grasp of a low_risk item only costs 15 points and you keep going. So for a high_risk target, being wrong is much more expensive than spending 10 on a human.
An observation is evidence of the room AT THAT MOMENT, not a guarantee: the roommate may move the item on the very next step. Therefore, after you observe a high_risk target and it was there, you may still choose to pick it (that is the cheap path) - but if its evidence is old, or you already had one failed grasp, ask_human instead: a human answer tells you where it is right now and the roommate leaves it alone until you grab it, so the answer cannot go stale.
A human answer refreshes only the named item and freezes it in place until you pick it up. A successful observe refreshes what is seen and marks a previously believed item absent if it should be in that room. After any missing pick, the ledger is updated to null: ask_human rather than retrying or guessing. Budget=60 is enough for careful verification; unnecessary asking wastes budget. If holding current target, place immediately."""

    def state_context(self) -> dict[str, Any]:
        context = super().state_context()
        context["versioned_state_ledger"] = self.ledger_snapshot()
        context["recent_history"] = self.history[-2:]
        return context

    def update(self, action, outcome) -> None:
        step = outcome["step"]
        if outcome.get("observation"):
            obs = outcome["observation"]
            self.ledger.observe(obs["room"], obs["items"], step)
        if action["action"] == "ask_human" and outcome.get("answer"):
            self.ledger.set_location(outcome["item"], outcome["location"], step, "ask_human")
        elif action["action"] == "pick":
            self.ledger.set_location(action["item"], "held_by_agent" if outcome["ok"] else None,
                                     step, "pick_success" if outcome["ok"] else "pick_absence")
        elif action["action"] == "place":
            self.ledger.set_location(action["item"], action["room"], step, "place")
        super().update(action, outcome)
