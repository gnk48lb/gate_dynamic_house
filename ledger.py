"""Versioned, overwrite-style world-state ledger; independent of the environment."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from env import ITEMS


class StateLedger:
    def __init__(self, inspection: dict[str, list[str]], inspection_gap: int):
        self.entries: dict[str, dict[str, Any]] = {}
        for room, items in inspection.items():
            for item in items:
                self._write(item, room, -inspection_gap, "initial_inspection")

    def _write(self, item: str, loc: str | None, step: int, source: str) -> None:
        previous = self.entries.get(item)
        history = previous["history"][:] if previous else []
        if previous:
            previous["status"] = "superseded"
            history.append({"loc": previous["loc"], "last_verified_step": previous["last_verified_step"],
                            "source": previous["source"], "status": "superseded",
                            "superseded_at_step": step})
        self.entries[item] = {"loc": loc, "last_verified_step": step, "status": "active",
                              "risk_level": ITEMS[item], "source": source, "history": history}

    def observe(self, room: str, items: list[str], step: int) -> None:
        observed = set(items)
        # A positive sighting supersedes the older location, even if the old room
        # was not observed. Absence is evidence only for the observed room.
        for item in items:
            entry = self.entries.get(item)
            if not entry or entry["status"] != "active" or entry["loc"] != room:
                self._write(item, room, step, "observe")
            else:
                entry["last_verified_step"] = step
                entry["source"] = "observe"
        for item, entry in list(self.entries.items()):
            if (entry["status"] == "active" and entry["loc"] == room
                    and item not in observed):
                self._write(item, None, step, "observe_absence")

    def set_location(self, item: str, loc: str | None, step: int, source: str) -> None:
        self._write(item, loc, step, source)

    def snapshot(self, current_step: int, include_history: bool = True) -> dict[str, dict[str, Any]]:
        """Compact snapshot for the agent prompt.

        The full version history stays in `entries` (and in the trace log) for
        auditing, but the prompt only carries what the decision actually needs:
        the active belief, its age, risk, and how many times it was superseded.
        Sending every historical location re-introduces exactly the stale-state
        confusion the ledger is meant to remove, and costs ~2x the tokens.

        One exception: when the active belief is null the item's location is
        unknown, and a null with no hint leaves the agent with no way to
        search. `last_loc` then names the room the item was last seen in, so
        the agent can search that room instead of wandering at random. It is a
        search hint about where to look next, not evidence for a pick.
        """
        result = {}
        for item, entry in self.entries.items():
            # `memory_age` already encodes `last_verified_step`, and a top-level
            # entry is by construction the active version, so neither is needed
            # in the prompt view.
            row = {"loc": entry["loc"], "memory_age": current_step - entry["last_verified_step"],
                   "risk_level": entry["risk_level"], "superseded_count": len(entry["history"])}
            if entry["loc"] is None and entry["history"]:
                last_seen = entry["history"][-1]["loc"]
                if last_seen is not None:
                    row["last_loc"] = last_seen
            result[item] = row
        return result

    def full_snapshot(self, current_step: int) -> dict[str, dict[str, Any]]:
        """Complete ledger including every version, for the audit trace."""
        result = {}
        for item, entry in self.entries.items():
            row = deepcopy(entry)
            row["memory_age"] = current_step - entry["last_verified_step"]
            result[item] = row
        return result

    def active_location(self, item: str) -> str | None:
        entry = self.entries.get(item)
        return entry["loc"] if entry and entry["status"] == "active" else None
