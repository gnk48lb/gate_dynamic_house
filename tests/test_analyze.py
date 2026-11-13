import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_episode(model, p, method, seed, success, cost, stale, asks, infra=False):
    return {"model_label": model, "model": f"x/{model}", "disturbance_p": p, "method": method,
            "seed": seed, "success": success, "failure_reason": None if success else "max_steps",
            "terminal_reason": None if success else "max_steps", "initial_budget": 60,
            "remaining_budget": 60 - cost, "total_cost": cost, "score": 0, "steps": 10,
            "ask_count": asks, "pick_count": 4, "stale_pick_count": stale, "safety_count": 0,
            "infrastructure_failure": infra,
            "token_usage": {"total_tokens": 1000, "input_tokens": 800, "output_tokens": 200,
                            "requests": 10, "request_retries": 0, "parse_retries": 0, "parse_failures": 0},
            "events": [], "ground_truth": {}, "agent_actions": [], "inspection": {}, "targets": []}


class AnalyzeTests(unittest.TestCase):
    def _write(self, directory, rows, manifest=None):
        path = directory / "episodes.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        if manifest:
            (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_pooled_verdict_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = []
            for model in ("m1", "m2"):
                for p in (0.0, 0.2, 0.4):
                    for seed in range(10):
                        # ledger: always succeeds and is cheaper than always_observe
                        rows.append(make_episode(model, p, "ledger_cost_gate", seed, True, 12, 0, 1))
                        # append_only: fails often -> 20pp+ gap
                        rows.append(make_episode(model, p, "append_only_memory", seed, seed % 5 == 0, 20, 3, 0))
                        # always_observe: succeeds but costs more
                        rows.append(make_episode(model, p, "always_observe", seed, True, 25, 0, 2))
                        rows.append(make_episode(model, p, "sliding_window", seed, seed % 3 == 0, 18, 2, 0))
            manifest = {"run_id": "t", "episodes_per_cell": 10, "budget": 60, "max_steps": 30,
                        "inspection_gap": 8, "temperature": 0, "max_tokens": 1000, "created_at": "now",
                        "models": {"m1": "codearts/m1", "m2": "codearts/m2"}}
            episodes = self._write(out, rows, manifest)
            result = subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--input", str(episodes),
                                     "--output-dir", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = (out / "summary.md").read_text(encoding="utf-8")
            self.assertIn("主闸门结论", summary)
            self.assertIn("通过", summary)
            self.assertIn("codearts/m1", summary)
            self.assertTrue((out / "pareto_frontier.png").exists())
            self.assertGreater((out / "pareto_frontier.png").stat().st_size, 5000)
            self.assertTrue((out / "metrics.csv").exists())
            verdicts = json.loads((out / "gate_verdicts.json").read_text(encoding="utf-8"))
            self.assertEqual(len(verdicts["primary_pooled_verdicts"]), 2)
            for verdict in verdicts["primary_pooled_verdicts"]:
                self.assertTrue(verdict["passed"])
                self.assertGreaterEqual(verdict["success_delta_pp"], 10)
                self.assertLess(verdict["paired_test"]["wilcoxon_p"], 0.05)
            print(result.stdout)

    def test_infrastructure_failures_are_a_sensitivity_check_not_a_veto(self):
        """Provider faults count as failures; the verdict only flips if dropping
        them would change the direction of the conclusion."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = []
            for model in ("m1",):
                for p in (0.2, 0.4):
                    for seed in range(10):
                        rows.append(make_episode(model, p, "ledger_cost_gate", seed, True, 12, 0, 1))
                        rows.append(make_episode(model, p, "append_only_memory", seed, seed % 5 == 0, 20, 3, 0))
                        rows.append(make_episode(model, p, "always_observe", seed, True, 25, 0, 2))
            # 5 of 10 ledger episodes failed on infrastructure
            for seed in range(5):
                rows.append(make_episode("m1", 0.2, "ledger_cost_gate", seed, False, 0, 0, 0, infra=True))
            episodes = self._write(out, rows)
            result = subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--input", str(episodes),
                                     "--output-dir", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            verdicts = json.loads((out / "gate_verdicts.json").read_text(encoding="utf-8"))
            by_p = {v["disturbance_p"]: v for v in verdicts["primary_pooled_verdicts"]}
            self.assertGreater(by_p[0.2]["infrastructure_failure_share"], 0.02)
            # The conclusion survives dropping the failed episodes, so it passes.
            self.assertTrue(by_p[0.2]["passed"])
            self.assertGreater(by_p[0.2]["success_delta_pp_excluding_infra"], 0)
            self.assertGreater(by_p[0.2]["cost_reduction_excluding_infra"], 0)

    def test_verdict_is_refused_when_dropping_failures_flips_the_direction(self):
        """If the infrastructure failures are what create the gap, refuse to judge."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = []
            for model in ("m1",):
                for p in (0.2, 0.4):
                    for seed in range(10):
                        # Without infrastructure noise append_only already matches ledger.
                        rows.append(make_episode(model, p, "ledger_cost_gate", seed, True, 12, 0, 1))
                        rows.append(make_episode(model, p, "append_only_memory", seed, True, 12, 0, 1))
                        rows.append(make_episode(model, p, "always_observe", seed, True, 25, 0, 2))
                    # The gap comes only from append_only failing on infrastructure.
                    for seed in range(5):
                        rows.append(make_episode(model, p, "append_only_memory", seed, False, 0, 0, 0, infra=True))
            episodes = self._write(out, rows)
            result = subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--input", str(episodes),
                                     "--output-dir", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            verdicts = json.loads((out / "gate_verdicts.json").read_text(encoding="utf-8"))
            for verdict in verdicts["primary_pooled_verdicts"]:
                self.assertFalse(verdict["passed"])
                self.assertIn("方向", verdict["reason"])


if __name__ == "__main__":
    unittest.main()
