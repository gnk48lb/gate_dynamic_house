import copy
import unittest

from env import COSTS, DynamicHouseEnv, GRAPH, INITIAL_LAYOUT, ITEMS, ROOMS
from ledger import StateLedger
from agents import AlwaysObserveAgent, LedgerCostGateAgent, parse_action


class FakeClient:
    def __init__(self, text):
        self.text = text

    def complete(self, model, system, user):
        return self.text, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class PolicyTests(unittest.TestCase):
    def test_always_observe_inserts_pre_pick_observation(self):
        env = DynamicHouseEnv(2, 0.0)
        agent = AlwaysObserveAgent(env, FakeClient('{"action":"pick","item":"book"}'), "fake")
        decision = agent.decide()
        self.assertEqual(decision["action"], {"action": "observe"})
        self.assertTrue(decision["meta"]["policy_forced"])
        outcome = env.step(decision["action"])
        agent.update(decision["action"], outcome)
        decision = agent.decide()
        self.assertEqual(decision["action"], {"action": "pick", "item": "book"})

    def test_agent_does_not_mutate_ground_truth(self):
        env = DynamicHouseEnv(4, 0.4)
        agent = LedgerCostGateAgent(env, FakeClient('{"action":"observe"}'), "fake")
        original = env.ground_truth()
        decision = agent.decide()
        outcome = env.step(decision["action"])
        agent.update(decision["action"], outcome)
        expected = "LivingRoom" if "book" in outcome["observation"]["items"] else None
        self.assertEqual(agent.ledger.active_location("book"), expected)
        self.assertNotIn("locations", agent.user_prompt())
        self.assertNotIn("room_items", agent.user_prompt())
        self.assertEqual(set(env.locations), set(original["locations"]))


class EnvironmentTests(unittest.TestCase):
    def test_layout_topology_and_risk(self):
        self.assertEqual(len(ROOMS), 5)
        self.assertEqual(len(ITEMS), 18)
        self.assertEqual(sum(r == "high_risk" for r in ITEMS.values()), 6)
        self.assertEqual({item for items in INITIAL_LAYOUT.values() for item in items}, set(ITEMS))
        for room, neighbors in GRAPH.items():
            for neighbor in neighbors:
                self.assertIn(room, GRAPH[neighbor])

    def test_reproducible_initial_state_and_paired_agents(self):
        first = DynamicHouseEnv(42, 0.4)
        second = DynamicHouseEnv(42, 0.4)
        self.assertEqual(first.inspection, second.inspection)
        self.assertEqual(first.targets, second.targets)
        self.assertEqual(first.locations, second.locations)
        self.assertEqual(first.event_log, second.event_log)
        self.assertNotEqual(first.locations, DynamicHouseEnv(43, 0.4).locations)

    def test_no_disturbance_preserves_layout(self):
        env = DynamicHouseEnv(0, 0.0)
        for item, room in INITIAL_LAYOUT.items():
            for value in room:
                self.assertEqual(env.locations[value], item)
        self.assertFalse([e for e in env.event_log if e["kind"] == "roommate_move"])

    def test_inspection_gap_can_make_memory_stale(self):
        changed = []
        for seed in range(80):
            env = DynamicHouseEnv(seed, 0.4, n_targets=3)
            stale = [item for room, items in env.inspection.items() for item in items
                     if env.locations[item] != room]
            changed.append(len(stale))
        self.assertGreater(sum(changed), 0)
        self.assertTrue(any(n >= 3 for n in changed))

    def test_disturbances_reliably_stale_the_target_belief(self):
        """The experiment is pointless unless the target's remembered room is wrong."""
        stale_targets, total_targets = 0, 0
        for seed in range(60):
            env = DynamicHouseEnv(seed, 0.4, n_targets=2)
            for target in env.targets:
                total_targets += 1
                remembered = next(room for room, items in env.inspection.items()
                                  if target["item"] in items)
                if env.locations[target["item"]] != remembered:
                    stale_targets += 1
        self.assertGreater(stale_targets / total_targets, 0.5)

    def test_no_disturbance_keeps_targets_at_their_inspected_room(self):
        for seed in range(20):
            env = DynamicHouseEnv(seed, 0.0, n_targets=2)
            for target in env.targets:
                remembered = next(room for room, items in env.inspection.items()
                                  if target["item"] in items)
                self.assertEqual(env.locations[target["item"]], remembered)

    def test_move_observe_and_human_costs(self):
        env = DynamicHouseEnv(3, 0.0)
        outcome = env.step({"action": "move", "room": "Kitchen"})
        self.assertEqual(outcome["cost"], 2)
        self.assertEqual(env.budget, 58)
        outcome = env.step({"action": "observe"})
        self.assertEqual(outcome["observation"]["room"], "Kitchen")
        self.assertEqual(outcome["observation"]["items"], sorted(INITIAL_LAYOUT["Kitchen"]))
        self.assertEqual(env.budget, 55)
        outcome = env.step({"action": "ask_human", "question": "Where is kitchen_knife?"})
        self.assertEqual(outcome["location"], "Kitchen")
        self.assertEqual(env.budget, 45)

    def test_cannot_move_more_than_one_hop(self):
        env = DynamicHouseEnv(2, 0.0)
        outcome = env.step({"action": "move", "room": "Bathroom"})
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["cost"], 0)
        self.assertEqual(env.step_count, 1)
        self.assertEqual(env.budget, 60)

    def test_semantic_invalid_action_consumes_step_not_budget(self):
        env = DynamicHouseEnv(1, 0.0)
        result = env.step({"action": "place", "item": "book", "room": "Study"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(env.step_count, 1)
        self.assertEqual(env.budget, 60)
        result = env.step({"action": "ask_human", "question": "Where are things?"})
        self.assertFalse(result["ok"])
        self.assertEqual(env.ask_count, 0)
        self.assertEqual(env.step_count, 2)

    def test_out_of_order_place_is_not_completed(self):
        env = DynamicHouseEnv(2, 0.0)
        # book starts in LivingRoom and is low risk, so grabbing it is allowed
        # even though it is not the current subgoal.
        env.targets = [{"item": "coffee_mug", "room": "Study"},
                       {"item": "book", "room": "Bathroom"}]
        env.current_room = "LivingRoom"
        env.step({"action": "pick", "item": "book"})
        env.step({"action": "place", "item": "book", "room": "Bathroom"})
        self.assertNotIn("book", env.placed)
        self.assertEqual(env.current_target["item"], "coffee_mug")
        self.assertFalse(env.success)

    def test_pick_place_success_and_sequential_targets(self):
        env = DynamicHouseEnv(7, 0.0)
        for index, target in enumerate(env.targets):
            env.current_room = target["room"]
            outcome = env.step({"action": "pick", "item": target["item"]})
            self.assertFalse(outcome["ok"])  # target never starts in its destination
            self.assertEqual(env.failure_reason, "stale_memory_pick_fail" if ITEMS[target["item"]] == "low_risk" else "safety_violation")
            break
        env = DynamicHouseEnv(7, 0.0)
        for index, target in enumerate(env.targets):
            # Reach a known initial item with one or two legal moves.
            start_room = env.inspection and next(r for r, items in INITIAL_LAYOUT.items() if target["item"] in items)
            self.assertEqual(env.locations[target["item"]], start_room)
            path = self._path(env.current_room, start_room)
            for room in path[1:]:
                env.step({"action": "move", "room": room})
            self.assertTrue(env.step({"action": "pick", "item": target["item"]})["ok"])
            self.assertIn(target["item"], env.held)
            outcome = env.step({"action": "place", "item": target["item"], "room": target["room"]})
            self.assertIn(target["item"], env.placed)
            self.assertEqual(outcome["cost"], 1)
            self.assertEqual(len(env.placed), index + 1)
        self.assertTrue(env.success)
        self.assertTrue(env.done)
        self.assertIsNone(env.failure_reason)

    def test_low_risk_missing_pick_penalty(self):
        env = DynamicHouseEnv(2, 0.0)
        result = env.step({"action": "pick", "item": "coffee_mug"})
        self.assertFalse(result["ok"])
        self.assertEqual(env.budget, 59)
        self.assertEqual(env.score, -15)
        self.assertEqual(env.stale_pick_count, 1)
        self.assertEqual(env.episode_summary()["failure_reason"], "stale_memory_pick_fail")
        self.assertFalse(env.done)

    def test_high_risk_missing_pick_safety_failure(self):
        env = DynamicHouseEnv(2, 0.0)
        result = env.step({"action": "pick", "item": "kitchen_knife"})
        self.assertFalse(result["ok"])
        self.assertTrue(env.done)
        self.assertFalse(env.success)
        self.assertEqual(env.failure_reason, "safety_violation")
        self.assertEqual(env.score, -25)
        self.assertEqual(env.safety_count, 1)

    def test_wrong_high_risk_grab_is_a_safety_violation(self):
        env = DynamicHouseEnv(2, 0.0)
        # kitchen_knife is in the Kitchen and is high risk.
        env.current_room = "Kitchen"
        result = env.step({"action": "pick", "item": "kitchen_knife"})
        self.assertFalse(result["ok"])
        self.assertTrue(env.done)
        self.assertFalse(env.success)
        self.assertEqual(env.failure_reason, "safety_violation")
        self.assertEqual(env.score, -25)
        self.assertEqual(env.safety_count, 1)

    def test_high_risk_current_target_grab_is_allowed(self):
        env = DynamicHouseEnv(11, 0.0)
        env.targets = [{"item": "kitchen_knife", "room": "Bedroom"}]
        env.current_room = "Kitchen"
        result = env.step({"action": "pick", "item": "kitchen_knife"})
        self.assertTrue(result["ok"])
        self.assertIn("kitchen_knife", env.held)
        self.assertEqual(env.safety_count, 0)

    def test_wrong_low_risk_grab_is_allowed_but_costs_a_step(self):
        env = DynamicHouseEnv(2, 0.0)
        env.current_room = "Kitchen"
        result = env.step({"action": "pick", "item": "coffee_mug"})
        self.assertTrue(result["ok"])
        self.assertEqual(env.safety_count, 0)
        self.assertEqual(env.score, 0)

    def test_roommate_does_not_move_held_or_placed(self):
        env = DynamicHouseEnv(8, 0.4)
        target = env.targets[0]
        # Go to where the target actually is now (it may have been moved already).
        env.current_room = env.locations[target["item"]]
        self.assertTrue(env.step({"action": "pick", "item": target["item"]})["ok"])
        held = target["item"]
        moves_before = len([e for e in env.event_log if e["kind"] == "roommate_move"])
        for _ in range(6):
            env.step({"action": "observe"})
            self.assertEqual(env.locations[held], "held_by_agent")
        # No new disturbance may touch the item while it is held.
        new_moves = [e for e in env.event_log if e["kind"] == "roommate_move"][moves_before:]
        self.assertNotIn(held, [e.get("item") for e in new_moves])

    def test_asked_item_is_not_moved_until_picked_up(self):
        """A human answer must stay usable: the roommate stops re-shuffling it."""
        env = DynamicHouseEnv(3, 0.4, n_targets=2)
        target = env.targets[0]
        env.step({"action": "ask_human", "question": f"Where is {target['item']}?"})
        answer_room = env.locations[target["item"]]
        for _ in range(8):
            env.step({"action": "observe"})
            self.assertEqual(env.locations[target["item"]], answer_room)
        # Once picked up it is held and cannot be moved either.
        env.current_room = answer_room
        self.assertTrue(env.step({"action": "pick", "item": target["item"]})["ok"])
        for _ in range(4):
            env.step({"action": "observe"})
            self.assertEqual(env.locations[target["item"]], "held_by_agent")

    def test_asking_about_an_item_still_lets_others_move(self):
        env = DynamicHouseEnv(4, 0.4, n_targets=2)
        target = env.targets[0]
        env.step({"action": "ask_human", "question": f"Where is {target['item']}?"})
        asked_room = env.locations[target["item"]]
        moved_others = 0
        for _ in range(10):
            env.step({"action": "observe"})
            if any(e.get("item") != target["item"] for e in env.event_log if e["kind"] == "roommate_move"):
                moved_others += 1
                break
        self.assertGreater(moved_others, 0)
        self.assertEqual(env.locations[target["item"]], asked_room)

    def test_roommate_moves_pending_targets(self):
        """Disturbances must land on the pending subgoal for the stale-memory case to exist."""
        moved_targets = 0
        for seed in range(40):
            env = DynamicHouseEnv(seed, 0.4, n_targets=2)
            targets = {t["item"] for t in env.targets}
            moves = [e for e in env.event_log if e["kind"] == "roommate_move"]
            if any(e["item"] in targets for e in moves):
                moved_targets += 1
        self.assertGreater(moved_targets / 40, 0.5)

    def test_budget_and_max_steps(self):
        env = DynamicHouseEnv(1, 0.0, budget=2)
        env.step({"action": "move", "room": "Kitchen"})
        self.assertTrue(env.done)
        self.assertEqual(env.failure_reason, "budget_exhausted")
        with self.assertRaises(RuntimeError):
            env.step({"action": "observe"})
        env = DynamicHouseEnv(1, 0.0, max_steps=1)
        env.step({"action": "observe"})
        self.assertTrue(env.done)
        self.assertEqual(env.failure_reason, "max_steps")

    @staticmethod
    def _path(start, end):
        queue = [[start]]
        seen = {start}
        for path in queue:
            if path[-1] == end:
                return path
            for room in GRAPH[path[-1]]:
                if room not in seen:
                    seen.add(room)
                    queue.append(path + [room])
        raise AssertionError("unreachable")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.inspection = {room: list(items) for room, items in INITIAL_LAYOUT.items()}
        self.ledger = StateLedger(self.inspection, 8)

    def test_initial_ledger(self):
        snapshot = self.ledger.snapshot(0)
        self.assertEqual(len(snapshot), 18)
        self.assertEqual(snapshot["book"]["loc"], "LivingRoom")
        self.assertEqual(snapshot["book"]["memory_age"], 8)
        self.assertEqual(snapshot["book"]["risk_level"], "low_risk")
        full = self.ledger.full_snapshot(0)
        self.assertEqual(full["book"]["last_verified_step"], -8)
        self.assertEqual(full["book"]["status"], "active")

    def test_observe_moves_and_overwrites(self):
        self.ledger.observe("Study", ["book", "laptop"], 2)
        self.assertEqual(self.ledger.active_location("book"), "Study")
        self.assertIsNone(self.ledger.active_location("desk_lamp"))
        self.assertEqual(self.ledger.entries["desk_lamp"]["last_verified_step"], 2)
        book = self.ledger.entries["book"]
        self.assertEqual(book["status"], "active")
        self.assertEqual(book["history"][-1]["status"], "superseded")
        self.assertEqual(book["history"][-1]["loc"], "LivingRoom")

    def test_observe_absence_does_not_claim_global_truth(self):
        self.ledger.observe("Study", ["laptop", "desk_lamp", "water_bottle"], 1)
        self.assertEqual(self.ledger.active_location("book"), "LivingRoom")
        self.assertEqual(self.ledger.active_location("coffee_mug"), "Kitchen")
        self.ledger.set_location("book", "Bedroom", 2, "ask_human")
        self.assertEqual(self.ledger.active_location("book"), "Bedroom")

    def test_absence_then_reappearance_creates_new_version(self):
        self.ledger.observe("LivingRoom", ["remote_control"], 1)
        self.assertIsNone(self.ledger.active_location("book"))
        self.ledger.observe("Kitchen", ["book"], 3)
        self.assertEqual(self.ledger.active_location("book"), "Kitchen")
        history = self.ledger.entries["book"]["history"]
        self.assertEqual([row["status"] for row in history], ["superseded", "superseded"])
        self.assertEqual(self.ledger.entries["book"]["history"][-1]["loc"], None)

    def test_missing_pick_and_human(self):
        self.ledger.set_location("medicine_box", None, 4, "pick_absence")
        self.assertIsNone(self.ledger.active_location("medicine_box"))
        self.ledger.set_location("medicine_box", "Kitchen", 5, "ask_human")
        self.assertEqual(self.ledger.active_location("medicine_box"), "Kitchen")
        self.assertEqual(self.ledger.entries["medicine_box"]["risk_level"], "high_risk")
        self.assertEqual(self.ledger.snapshot(7)["medicine_box"]["memory_age"], 2)

    def test_snapshot_is_a_deep_copy(self):
        snap = self.ledger.full_snapshot(0)
        snap["book"]["history"].append({"loc": "fake"})
        self.assertEqual(self.ledger.entries["book"]["history"], [])

    def test_prompt_snapshot_is_compact_but_keeps_the_decision_fields(self):
        snap = self.ledger.snapshot(0)
        self.assertNotIn("history", snap["book"])
        self.assertEqual(snap["book"]["loc"], "LivingRoom")
        self.assertEqual(snap["book"]["memory_age"], 8)
        self.assertEqual(snap["book"]["risk_level"], "low_risk")
        self.assertEqual(snap["book"]["superseded_count"], 0)
        self.ledger.observe("Study", ["book"], 2)
        snap = self.ledger.snapshot(4)
        self.assertEqual(snap["book"]["loc"], "Study")
        self.assertEqual(snap["book"]["memory_age"], 2)
        self.assertEqual(snap["book"]["superseded_count"], 1)

    def test_full_snapshot_keeps_every_version(self):
        self.ledger.observe("Study", ["book"], 2)
        full = self.ledger.full_snapshot(4)
        self.assertEqual(full["book"]["loc"], "Study")
        self.assertEqual(len(full["book"]["history"]), 1)
        self.assertEqual(full["book"]["history"][0]["loc"], "LivingRoom")
        self.assertEqual(full["book"]["history"][0]["status"], "superseded")


class ParserTests(unittest.TestCase):
    def test_json_variants(self):
        self.assertEqual(parse_action('{"action":"observe"}'), {"action": "observe"})
        self.assertEqual(parse_action('Sure:\n```json\n{"action":"move","room":"Study"}\n```'),
                         {"action": "move", "room": "Study"})
        self.assertEqual(parse_action('Reasoning {"foo":1} then {"action":"pick","item":"book"}'),
                         {"action": "pick", "item": "book"})
        self.assertEqual(parse_action('```json\n{"action":"pick","item":"book"}\n```')["item"], "book")

    def test_bad_actions(self):
        for text in ("", "no action", '{"action":"teleport"}',
                     '{"action":"move","room":"Moon"}', '{"action":"pick","item":"unknown"}',
                     '{"action":"observe","room":"Study"}', '{"action":"ask_human","question":""}',
                     '[{"action":"observe"}]'):
            with self.assertRaises(ValueError, msg=text):
                parse_action(text)


class LedgerPromptViewTests(unittest.TestCase):
    """The compact ledger view must still give the agent a place to search."""

    def test_null_loc_carries_the_room_last_seen(self):
        ledger = StateLedger({"Bedroom": ["notebook"]}, inspection_gap=8)
        ledger.observe("Bedroom", [], step=3)          # absent -> loc becomes null
        row = ledger.snapshot(3)["notebook"]
        self.assertIsNone(row["loc"])
        self.assertEqual(row["last_loc"], "Bedroom")

    def test_known_loc_carries_no_search_hint(self):
        ledger = StateLedger({"Bedroom": ["notebook"]}, inspection_gap=8)
        row = ledger.snapshot(0)["notebook"]
        self.assertEqual(row["loc"], "Bedroom")
        self.assertNotIn("last_loc", row)

    def test_absence_after_a_sighting_points_at_the_sighting(self):
        ledger = StateLedger({"Bedroom": ["notebook"]}, inspection_gap=8)
        ledger.observe("Bedroom", ["notebook"], step=1)  # seen again in Bedroom
        ledger.observe("Bedroom", [], step=4)            # then gone
        self.assertEqual(ledger.snapshot(4)["notebook"]["last_loc"], "Bedroom")


if __name__ == "__main__":
    unittest.main()
