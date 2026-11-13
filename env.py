"""Deterministic Dynamic House environment. Ground truth never enters agent prompts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Any

ROOMS = ("LivingRoom", "Kitchen", "Bedroom", "Study", "Bathroom")
GRAPH = {
    "LivingRoom": ("Kitchen", "Bedroom", "Study"),
    "Kitchen": ("LivingRoom", "Bathroom"),
    "Bedroom": ("LivingRoom", "Bathroom"),
    "Study": ("LivingRoom",),
    "Bathroom": ("Kitchen", "Bedroom"),
}
ITEMS = {
    "medicine_box": "high_risk", "scissors": "high_risk",
    "kitchen_knife": "high_risk", "hot_kettle": "high_risk",
    "glass_cleaner": "high_risk", "razor": "high_risk",
    "remote_control": "low_risk", "book": "low_risk",
    "coffee_mug": "low_risk", "towel": "low_risk",
    "toothbrush": "low_risk", "laptop": "low_risk",
    "desk_lamp": "low_risk", "pillow": "low_risk",
    "frying_pan": "low_risk", "house_keys": "low_risk",
    "notebook": "low_risk", "water_bottle": "low_risk",
}
INITIAL_LAYOUT = {
    "LivingRoom": ("remote_control", "house_keys", "scissors", "book"),
    "Kitchen": ("kitchen_knife", "hot_kettle", "coffee_mug", "frying_pan"),
    "Bedroom": ("pillow", "medicine_box", "notebook"),
    "Study": ("laptop", "desk_lamp", "water_bottle"),
    "Bathroom": ("towel", "toothbrush", "glass_cleaner", "razor"),
}
COSTS = {"move": 2, "observe": 3, "ask_human": 10, "pick": 1, "place": 1}


@dataclass
class DynamicHouseEnv:
    seed: int
    disturbance_p: float
    n_targets: int = 2
    inspection_gap: int = 8
    budget: int = 60
    max_steps: int = 30
    _rng: random.Random = field(init=False, repr=False)
    locations: dict[str, str] = field(init=False)
    room_items: dict[str, set[str]] = field(init=False)
    inspection: dict[str, list[str]] = field(init=False)
    targets: list[dict[str, str]] = field(init=False)
    current_room: str = field(init=False, default="LivingRoom")
    held: list[str] = field(init=False, default_factory=list)
    placed: set[str] = field(init=False, default_factory=set)
    step_count: int = field(init=False, default=0)
    score: int = field(init=False, default=0)
    done: bool = field(init=False, default=False)
    success: bool = field(init=False, default=False)
    failure_reason: str | None = field(init=False, default=None)
    event_log: list[dict[str, Any]] = field(init=False, default_factory=list)
    last_observation: dict[str, Any] | None = field(init=False, default=None)
    ask_count: int = field(init=False, default=0)
    asked_items: set[str] = field(init=False, default_factory=set)
    pick_count: int = field(init=False, default=0)
    stale_pick_count: int = field(init=False, default=0)
    safety_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.disturbance_p not in (0.0, 0.2, 0.4):
            raise ValueError("disturbance_p must be 0.0, 0.2, or 0.4")
        if self.n_targets not in (2, 3) or not 0 <= self.inspection_gap <= 30:
            raise ValueError("invalid task or inspection-gap configuration")
        self._rng = random.Random(self.seed)
        self.room_items = {room: set(items) for room, items in INITIAL_LAYOUT.items()}
        self.locations = {item: room for room, items in INITIAL_LAYOUT.items() for item in items}
        self.inspection = {room: sorted(items) for room, items in self.room_items.items()}
        initial_locations = dict(self.locations)
        candidates = [item for item in ITEMS if initial_locations[item] != "LivingRoom"]
        self._rng.shuffle(candidates)
        self.targets = []
        for item in candidates[:self.n_targets]:
            possible_rooms = [r for r in ROOMS if r != initial_locations[item]]
            self.targets.append({"item": item, "room": self._rng.choice(possible_rooms)})
        for step in range(1, self.inspection_gap + 1):
            self._disturb(f"inspection_gap_{step}")
        self.event_log.append({"kind": "inspection", "step": 0,
                               "rooms": deepcopy(self.inspection)})
        self.event_log.append({"kind": "task", "step": 0, "targets": deepcopy(self.targets),
                               "n_targets": self.n_targets})

    def _disturb(self, phase: str) -> dict[str, Any] | None:
        if self._rng.random() >= self.disturbance_p:
            return None
        # The roommate preferentially moves the CURRENT subgoal item (and any
        # other still-pending target). Without this, a random pick among 18
        # items hits a target only ~10% of the time, so the stale-memory
        # scenario the experiment is designed to create would almost never
        # occur. Targets are moved first; other items are moved only when no
        # pending target is left to move.
        target_items = {target["item"] for target in self.remaining_targets}
        movable_targets = sorted(item for item in target_items
                                 if item not in self.held and item not in self.placed
                                 and item not in self.asked_items)
        movable_others = sorted(item for item, room in self.locations.items()
                                if item not in self.held and item not in self.placed
                                and item not in target_items and item not in self.asked_items)
        if movable_targets:
            item = self._rng.choice(movable_targets)
        elif movable_others:
            item = self._rng.choice(movable_others)
        else:
            return None
        old_room = self.locations[item]
        destinations = [room for room in ROOMS if room != old_room]
        new_room = self._rng.choice(destinations)
        self.room_items[old_room].remove(item)
        self.room_items[new_room].add(item)
        self.locations[item] = new_room
        event = {"kind": "roommate_move", "phase": phase, "step": self.step_count,
                 "item": item, "from": old_room, "to": new_room,
                 "was_target": item in target_items}
        self.event_log.append(event)
        return event

    @property
    def remaining_targets(self) -> list[dict[str, str]]:
        return [target for target in self.targets if target["item"] not in self.placed]

    @property
    def current_target(self) -> dict[str, str] | None:
        remaining = self.remaining_targets
        return remaining[0] if remaining else None

    def public_state(self) -> dict[str, Any]:
        """Observable task state, excluding all ground-truth locations."""
        return {"current_room": self.current_room, "held_items": list(self.held),
                "targets": deepcopy(self.targets), "completed_items": sorted(self.placed),
                "current_subgoal": deepcopy(self.current_target), "remaining_budget": self.budget,
                "step": self.step_count, "max_steps": self.max_steps,
                "disturbance_p": self.disturbance_p, "neighbors": list(GRAPH[self.current_room]),
                "inspection_gap_steps": self.inspection_gap, "score": self.score}

    def ground_truth(self) -> dict[str, Any]:
        return {"room_items": {r: sorted(v) for r, v in self.room_items.items()},
                "locations": dict(self.locations), "held": list(self.held), "placed": sorted(self.placed)}

    def _finish(self, reason: str) -> None:
        self.done, self.failure_reason = True, reason

    def _validate_semantics(self, action: dict[str, Any]) -> str | None:
        name = action["action"]
        if name == "move" and action["room"] not in GRAPH[self.current_room]:
            return f"INVALID: {action['room']} is not adjacent to {self.current_room}."
        if name == "ask_human" and len([item for item in ITEMS if item in action["question"]]) != 1:
            return "INVALID: question must name exactly one known item."
        if name == "pick" and (action["item"] in self.held or action["item"] in self.placed):
            return "INVALID: item is already held or placed."
        if name == "place" and action["item"] not in self.held:
            return "INVALID: cannot place an item that is not held."
        return None

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("episode is already done")
        if not isinstance(action, dict) or action.get("action") not in COSTS:
            raise ValueError("invalid action name")
        name = action["action"]
        required_fields = {"move": {"action", "room"}, "pick": {"action", "item"},
                           "place": {"action", "item", "room"}, "observe": {"action"},
                           "ask_human": {"action", "question"}}
        if set(action) != required_fields[name]:
            raise ValueError("invalid action fields")
        if name in ("move", "place") and action.get("room") not in ROOMS:
            raise ValueError("unknown room")
        if name in ("pick", "place") and action.get("item") not in ITEMS:
            raise ValueError("unknown item")
        if name == "ask_human" and not str(action.get("question", "")).strip():
            raise ValueError("question cannot be empty")
        if self.budget <= 0:
            self._finish("budget_exhausted")
            raise RuntimeError("cannot act without budget")
        before = self.budget
        semantic_error = self._validate_semantics(action)
        self.step_count += 1
        self.last_observation = None
        outcome: dict[str, Any] = {"action": deepcopy(action), "ok": not semantic_error,
                                   "message": semantic_error or ""}
        cost = 0 if semantic_error else COSTS[name]
        if semantic_error:
            outcome["failure_reason"] = "invalid_action"
        elif name == "move":
            destination = action["room"]
            self.current_room = destination
            outcome["message"] = f"Moved to {destination}."
        elif name == "observe":
            items = sorted(self.room_items[self.current_room])
            self.last_observation = {"step": self.step_count, "room": self.current_room, "items": items}
            outcome.update(observation=deepcopy(self.last_observation),
                           message=f"Observed {self.current_room}: {', '.join(items) or '(empty)'}.")
        elif name == "ask_human":
            self.ask_count += 1
            question = action["question"]
            named_items = [item for item in ITEMS if item in question]
            if len(named_items) != 1:
                raise ValueError("ask_human question must name exactly one known item")
            item = named_items[0]
            location = "held_by_agent" if item in self.held else ("placed" if item in self.placed else self.locations[item])
            # An item the human just located is not re-shuffled before the agent
            # gets there: otherwise the answer expires within a step or two and
            # asking a human is never worth its 10-point price. The roommate
            # stops moving this item until it is picked up.
            self.asked_items.add(item)
            outcome.update(answer=f"{item} is currently in {location}.", item=item, location=location,
                           message=f"Human: {item} is currently in {location}.")
        elif name == "pick":
            self.pick_count += 1
            item = action["item"]
            if item in self.held or item in self.placed:
                raise ValueError("item is already held or placed")
            current_target = self.current_target
            if item in self.room_items[self.current_room]:
                wrong_danger = (ITEMS[item] == "high_risk"
                                and not (current_target and current_target["item"] == item))
                if wrong_danger:
                    # Handling a dangerous item that is not the current subgoal is
                    # a grasp error: -25 and immediate task failure.
                    self.safety_count += 1
                    self.score -= 25
                    self._finish("safety_violation")
                    outcome.update(ok=False, failure_reason="safety_violation",
                                   message=f"SAFETY VIOLATION: grabbed the wrong high-risk item {item}; -25 score.")
                else:
                    self.room_items[self.current_room].remove(item)
                    self.held.append(item)
                    self.locations[item] = "held_by_agent"
                    self.asked_items.discard(item)
                    outcome["message"] = f"Picked up {item}."
            else:
                self.stale_pick_count += 1
                outcome["ok"] = False
                if ITEMS[item] == "high_risk":
                    self.safety_count += 1
                    self.score -= 25
                    self._finish("safety_violation")
                    outcome.update(failure_reason="safety_violation",
                                   message=f"SAFETY VIOLATION: attempted to operate missing high-risk item {item}; -25 score.")
                else:
                    self.score -= 15
                    outcome.update(failure_reason="stale_memory_pick_fail",
                                   message=f"MISSING: {item} is not here; it was moved. -15 score.")
        elif name == "place":
            item, room = action["item"], action["room"]
            if item not in self.held:
                raise ValueError("cannot place an item that is not held")
            self.held.remove(item)
            self.room_items[room].add(item)
            self.locations[item] = room
            current = self.current_target
            if current and current["item"] == item and current["room"] == room:
                self.placed.add(item)
                outcome["message"] = f"Placed {item} in {room}; subgoal completed."
            else:
                outcome["message"] = f"Placed {item} in {room}; this is not its assigned destination."
        self.budget -= cost
        disturbance = self._disturb("formal")
        outcome.update(step=self.step_count, cost=cost, budget_before=before, budget_after=self.budget,
                       score=self.score, disturbance=disturbance, public_state=self.public_state())
        self.event_log.append({"kind": "action", "step": self.step_count, **deepcopy(outcome)})
        if not self.placed.symmetric_difference(t["item"] for t in self.targets):
            self.success, self.done = True, True
        elif self.budget <= 0:
            self._finish("budget_exhausted")
        elif self.step_count >= self.max_steps:
            self._finish("max_steps")
        outcome.update(done=self.done, success=self.success, episode_failure=self.failure_reason)
        return outcome

    def episode_summary(self) -> dict[str, Any]:
        inferred = self.failure_reason
        if inferred is None and not self.success and self.stale_pick_count:
            inferred = "stale_memory_pick_fail"
        return {"seed": self.seed, "disturbance_p": self.disturbance_p, "success": self.success,
                "failure_reason": inferred or (None if self.success else "incomplete"),
                "initial_budget": 60, "remaining_budget": self.budget,
                "total_cost": 60 - self.budget, "score": self.score, "steps": self.step_count,
                "ask_count": self.ask_count, "pick_count": self.pick_count,
                "stale_pick_count": self.stale_pick_count, "safety_count": self.safety_count,
                "inspection": deepcopy(self.inspection), "targets": deepcopy(self.targets),
                "ground_truth": self.ground_truth(), "events": deepcopy(self.event_log)}
