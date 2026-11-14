import json
from pathlib import Path
import tempfile
import unittest

from agents import LedgerCostGateAgent
from analyze import paired_cost_test
from env import DynamicHouseEnv
from run_benchmark import run_episode, select_repair_candidates, select_reusable_rows


class FakeClient:
    def complete(self, model, system, user):
        return '{"action":"observe"}', {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


class BenchmarkTests(unittest.TestCase):
    def test_episode_logs_have_paired_state_and_budget(self):
        summary, trace, raw = run_episode(1, 0.2, "ledger_cost_gate", "test", "fake", FakeClient(), 0, "test-run")
        self.assertFalse(summary["success"])
        self.assertEqual(summary["failure_reason"], "budget_exhausted")
        self.assertEqual(len(trace), 20)
        action_events = [e for e in summary["events"] if e["kind"] == "action"]
        disturbances = [e for e in summary["events"] if e["kind"] == "roommate_move"]
        self.assertEqual(len(action_events), 20)
        self.assertEqual(len(summary["events"]), len(action_events) + len(disturbances) + 2)
        for index, row in enumerate(trace):
            self.assertEqual(row["step"], index + 1)
            self.assertEqual(row["state_before_action"]["remaining_budget"], 60 - index * 3)
            self.assertEqual(row["public_state_after"]["remaining_budget"], 57 - index * 3)
            self.assertLessEqual(row["ledger_snapshot"]["book"]["memory_age"], row["step"] + 8)
        self.assertEqual(summary["token_usage"]["total_tokens"], 240)
        self.assertEqual(len(raw["responses"]), 20)

    def test_protocol_error_does_not_change_budget(self):
        summary, trace, raw = run_episode(1, 0.0, "ledger_cost_gate", "test", "fake", BadClient(), 1, "test-run")
        self.assertTrue(summary["infrastructure_failure"])
        self.assertEqual(summary["failure_reason"], "infrastructure_failure")
        self.assertEqual(summary["remaining_budget"], 60)
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(trace, [])
        self.assertEqual(len(raw["responses"]), 2)

    def test_statistics(self):
        result = paired_cost_test([3, 4, 5, 4, 5, 5])
        self.assertAlmostEqual(result["mean_cost_reduction"], 26 / 6)
        self.assertLess(result["wilcoxon_p"], 0.05)
        self.assertGreater(result["ci95"][0], 0)
        self.assertEqual(paired_cost_test([0, 0])["wilcoxon_p"], 1.0)


class RepairSelectionTests(unittest.TestCase):
    """The repair pass must not re-run cells that are already finished."""

    def _row(self, label="deepseek", model="workbuddy/deepseek-v4.1-flash", p=0.2, seed=0,
             method="ledger_cost_gate", infra=False):
        return {"model_label": label, "model": model, "disturbance_p": p, "seed": seed,
                "method": method, "infrastructure_failure": infra}

    def test_repair_uses_model_label_and_drops_superseded_failures(self):
        # The cell key must use the LABEL: the same model on a swapped route
        # still shares the cell. Once the repaired row lands, the cell's newest
        # row is no longer an infrastructure failure, so nothing is re-run.
        failed = self._row(infra=True)
        repaired = self._row(model="workbuddy-ai/deepseek-v4.1-flash-sg", infra=False)
        rows = [failed, repaired]
        self.assertEqual(select_repair_candidates(rows, set()), [])
        # Without the repair the failed row is still the newest one, and it is
        # re-run (it is not in `done`).
        self.assertEqual(len(select_repair_candidates([failed], set())), 1)

    def test_repair_skips_cells_already_in_done(self):
        row = self._row(infra=True)
        done = {("deepseek", 0.2, 0, "ledger_cost_gate")}
        self.assertEqual(select_repair_candidates([row], done), [])

    def test_repair_ignores_label_model_mismatch(self):
        # A label whose route changed is still the same cell: matching on the
        # raw model id would re-run an episode that is already accounted for.
        row = self._row(label="deepseek", model="workbuddy-ai/deepseek-v4.1-flash-sg", infra=True)
        done = {("deepseek", 0.2, 0, "ledger_cost_gate")}
        self.assertEqual(select_repair_candidates([row], done), [])


class ReuseSelectionTests(unittest.TestCase):
    """--reuse-from must not resurrect a method that is being re-run."""

    def _row(self, label, method, p=0.2, seed=0, infra=False):
        return {"model_label": label, "model": "m", "disturbance_p": p, "seed": seed,
                "method": method, "infrastructure_failure": infra}

    def test_rerun_method_is_not_reused(self):
        rows = [self._row("minimax", "ledger_cost_gate"), self._row("minimax", "append_only_memory")]
        kept = select_reusable_rows(rows, {"minimax": "m"}, set(), {"ledger_cost_gate"})
        self.assertEqual([row["method"] for row in kept], ["append_only_memory"])

    def test_other_models_and_infra_rows_are_not_reused(self):
        rows = [self._row("deepseek", "append_only_memory"),
                self._row("minimax", "append_only_memory", infra=True)]
        kept = select_reusable_rows(rows, {"minimax": "m"}, set(), set())
        self.assertEqual(kept, [])


class BadClient:
    def complete(self, model, system, user):
        return 'this is not JSON', {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


class DeadlineTests(unittest.TestCase):
    def test_hard_deadline_fires_on_a_hung_route(self):
        import time
        import urllib.request
        import agents

        original = urllib.request.urlopen

        def hung(*args, **kwargs):
            time.sleep(30)
            raise AssertionError("unreachable")

        urllib.request.urlopen = hung
        try:
            client = agents.MagpieClient(timeout=0.5, max_retries=0)
            started = time.monotonic()
            with self.assertRaises(RuntimeError) as caught:
                client.complete("fake", "sys", "user")
            self.assertIn("deadline", str(caught.exception))
            self.assertLess(time.monotonic() - started, 10)
        finally:
            urllib.request.urlopen = original


if __name__ == "__main__":
    unittest.main()
