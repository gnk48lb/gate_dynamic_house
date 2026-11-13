"""Concurrent, resumable Dynamic House LLM benchmark.

Examples (from this directory):
    python run_benchmark.py --episodes 5 --output-dir results_smoke --models deepseek
    python run_benchmark.py --episodes 80 --output-dir results --models deepseek,kimi,glm
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import threading
import time
import traceback
from typing import Any
import urllib.request

from agents import (AlwaysObserveAgent, AppendOnlyMemoryAgent, LedgerCostGateAgent,
                    MagpieClient, SlidingWindowAgent)
from env import DynamicHouseEnv

ROOT = Path(__file__).resolve().parent
AGENTS = {cls.name: cls for cls in (SlidingWindowAgent, AppendOnlyMemoryAgent,
                                    AlwaysObserveAgent, LedgerCostGateAgent)}
METHODS = tuple(AGENTS)
PS = (0.0, 0.2, 0.4)
print_lock = threading.Lock()


class JsonlWriter:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_catalog(base_url: str, api_key: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))["data"]


def resolve_models(base_url: str, api_key: str, requested: str) -> dict[str, str]:
    available = {entry["id"] for entry in model_catalog(base_url, api_key)}
    presets = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    names = list(presets) if requested == "all" else requested.split(",")
    models = {name.strip(): presets.get(name.strip(), name.strip()) for name in names if name.strip()}
    missing = {label: model for label, model in models.items() if model not in available}
    if missing:
        raise RuntimeError(f"Models not in live Magpie catalog: {missing}. Available: {sorted(available)}")
    return models


def run_episode(seed: int, p: float, method: str, model_label: str, model: str,
                client: MagpieClient, parse_retries: int, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    env = DynamicHouseEnv(seed, p, n_targets=2 + seed % 2)
    agent = AGENTS[method](env, client, model, parse_retries=parse_retries)
    started = time.monotonic()
    trace: list[dict[str, Any]] = []
    infrastructure_failure = None
    episode_id = f"{run_id}:{model_label}:p{p}:s{seed}:{method}"
    while not env.done:
        step = env.step_count + 1
        decision = None
        state_before = env.public_state()
        try:
            decision = agent.decide()
            outcome = env.step(decision["action"])
        except (RuntimeError, ValueError) as exc:
            infrastructure_failure = {"episode_id": episode_id, "model_label": model_label, "model": model,
                                      "disturbance_p": p, "seed": seed, "method": method,
                                      "step": step, "failure_type": "agent_protocol_or_transport",
                                      "error": str(exc), "time": now()}
            break
        agent.update(decision["action"], outcome)
        trace.append({"schema_version": 1, "run_id": run_id, "episode_id": episode_id,
                      "time": now(), "model_label": model_label, "model": model,
                      "disturbance_p": p, "seed": seed, "method": method, "step": step,
                      "state_before_action": state_before,
                      "public_state_after": outcome["public_state"], "action": decision["action"],
                      "action_outcome": {k: v for k, v in outcome.items() if k not in ("public_state", "disturbance")},
                      "ledger_snapshot": agent.full_ledger_snapshot(),
                      "policy_forced": decision["meta"].get("policy_forced", False),
                      "model_raw_response": decision.get("raw"),
                      "request": {k: v for k, v in decision["meta"].items() if k != "policy_forced"},
                      "ground_truth_after": env.ground_truth(), "roommate_event": outcome.get("disturbance"),
                      "token_usage_cumulative": deepcopy(agent.total_usage)})
    summary = env.episode_summary()
    terminal_reason = env.failure_reason
    final_failure = terminal_reason
    if terminal_reason is None and not env.success:
        final_failure = "max_steps"
    elif terminal_reason in ("max_steps", "budget_exhausted") and env.stale_pick_count:
        # The episode was derailed by acting on stale memory; keep the terminal
        # reason separately so the root cause stays visible in the failure mix.
        final_failure = "stale_memory_pick_fail"
    summary["terminal_reason"] = terminal_reason
    summary["failure_reason"] = final_failure or (None if env.success else "incomplete")
    summary.update(schema_version=1, run_id=run_id, episode_id=episode_id, time=now(),
                   model_label=model_label, model=model, method=method,
                   infrastructure_failure=infrastructure_failure is not None,
                   token_usage=agent.total_usage, wall_time_s=round(time.monotonic() - started, 3),
                   agent_actions=agent.action_log)
    if infrastructure_failure:
        summary["success"] = False
        summary["failure_reason"] = "infrastructure_failure"
        summary["infrastructure_error"] = infrastructure_failure["error"]
        summary["infrastructure_failure_detail"] = infrastructure_failure
    raw = {"schema_version": 1, "run_id": run_id, "episode_id": episode_id,
           "model_label": model_label, "model": model, "disturbance_p": p,
           "seed": seed, "method": method, "responses": agent.raw_responses}
    return summary, trace, raw


def provider_of(model: str) -> str:
    """The upstream account that serves a model; the unit of upstream capacity."""
    return model.split("/")[0] if "/" in model else model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=80, help="seeds per model x p x method")
    parser.add_argument("--models", default="deepseek,glm,minimax", help="preset names, explicit ids, or all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--base-url", default=os.getenv("MAGPIE_BASE_URL", "http://127.0.0.1:3425"))
    parser.add_argument("--api-key", default=os.getenv("MAGPIE_API_KEY", "local"))
    parser.add_argument("--concurrency", type=int, default=12,
                        help="upper bound on episodes in flight across all providers")
    parser.add_argument("--provider-concurrency", type=int, default=4,
                        help="max concurrent episodes per upstream provider account")
    parser.add_argument("--timeout", type=float, default=150,
                        help="hard wall-clock deadline per request; upstream soft-queueing "
                             "counts against it, so keep it well above the observed p99 latency")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="output budget per request; thinking models need room or they "
                             "return an empty body (finish_reason=length)")
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--repair-infrastructure", action="store_true", default=True,
                        help="re-run episodes that failed on provider errors, at low concurrency")
    parser.add_argument("--repair-concurrency", type=int, default=4)
    parser.add_argument("--retry-infrastructure", action="store_true",
                        help="re-run only the episodes that failed on provider errors")
    parser.add_argument("--model-alias", default="",
                        help="report the retried episodes under this model label instead of the "
                             "provider-specific one (keeps a model's results together when a "
                             "route is swapped mid-run)")
    parser.add_argument("--rerun-methods", default="",
                        help="comma-separated methods whose existing episodes must be discarded and re-run "
                             "(e.g. after a prompt change), keeping the rest of the matrix intact")
    parser.add_argument("--reuse-from", type=Path, default=None,
                        help="episodes.jsonl from an earlier run whose completed episodes should be reused")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.concurrency < 1 or args.seed_start < 0:
        parser.error("episodes/concurrency must be positive; seed-start nonnegative")
    methods = args.methods.split(",")
    if any(method not in AGENTS for method in methods):
        parser.error(f"unknown method; choose from {METHODS}")
    models = resolve_models(args.base_url, args.api_key, args.models)
    if args.model_alias:
        # Every requested preset is reported under one label: the route for that
        # model may have been swapped for the cells being retried.
        first_label, first_model = next(iter(models.items()))
        models = {args.model_alias: first_model}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_config = {"schema_version": 1, "episodes_per_cell": args.episodes,
                        "seed_start": args.seed_start, "models": models, "disturbance_levels": list(PS),
                        "methods": methods, "budget": 60, "max_steps": 30, "inspection_gap": 8,
                        "target_count_rule": "2 + seed % 2", "base_url": args.base_url,
                        "temperature": 0, "max_tokens": args.max_tokens, "timeout_s": args.timeout,
                        "request_retries": args.request_retries, "parse_retries": args.parse_retries}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        config = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_config = {k: v for k, v in config.items() if k not in ("run_id", "created_at")}
        # max_tokens and timeout are transport knobs, not experimental
        # conditions: changing them on a retry does not mix conditions.
        for knob in ("max_tokens", "timeout_s"):
            previous_config.pop(knob, None)
        comparable = {k: v for k, v in requested_config.items() if k not in ("max_tokens", "timeout_s")}
        if args.model_alias:
            # A retry may serve the same model through a different route and
            # may cover only a subset of the models, so compare labels as sets.
            comparable["models"] = sorted(comparable["models"])
            previous_config["models"] = sorted(previous_config.get("models", {}))
            if not set(comparable["models"]).issubset(set(previous_config["models"])):
                raise RuntimeError("Output manifest differs. Use a new output directory; refusing to mix configurations.")
            del comparable["models"]
            del previous_config["models"]
        if previous_config != comparable:
            raise RuntimeError(f"Output manifest differs. Use a new output directory; refusing to mix configurations. "
                               f"\n  previous: {previous_config}\n  requested: {comparable}")
        run_id = config["run_id"]
    else:
        run_id = args.run_id or time.strftime("dynamic_house_%Y%m%d_%H%M%S")
        config = {"run_id": run_id, "created_at": now(), **requested_config}
        manifest_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    episodes_path, traces_path = output / "episodes.jsonl", output / "traces.jsonl"
    raw_path, failures_path = output / "raw_responses.jsonl", output / "failures.jsonl"
    # A method whose prompt changed must be re-run from scratch; its old rows
    # stay in the log but the newest row for a cell wins (see analyze.py).
    rerun_methods = {m.strip() for m in args.rerun_methods.split(",") if m.strip()}
    # Keys use the model LABEL: a route may be swapped mid-run while the model
    # it serves stays the same.
    done = {(r["model_label"], r["disturbance_p"], r["seed"], r["method"]) for r in load_jsonl(episodes_path)}
    if rerun_methods:
        dropped = {key for key in done if key[3] in rerun_methods}
        done -= dropped
        print(f"Re-running {len(rerun_methods)} method(s) {sorted(rerun_methods)}: "
              f"{len(dropped)} existing episodes will be superseded", flush=True)
    # Only the episodes that failed on infrastructure may be retried: they are
    # provider faults, not policy outcomes.
    if args.retry_infrastructure:
        stale_infra = set()
        for row in load_jsonl(episodes_path):
            if row.get("infrastructure_failure"):
                stale_infra.add((row["model_label"], row["disturbance_p"], row["seed"], row["method"]))
        done -= stale_infra
        print(f"Retrying {len(stale_infra)} infrastructure-failed episodes", flush=True)
    # Episodes completed in an earlier run (e.g. a model was dropped from the
    # matrix) can be reused instead of being paid for twice.
    reused = 0
    if args.reuse_from:
        for row in load_jsonl(args.reuse_from):
            key = (row["model_label"], row["disturbance_p"], row["seed"], row["method"])
            if key in done or row["model_label"] not in models:
                continue
            if row.get("infrastructure_failure"):
                continue
            writers_episode = JsonlWriter(episodes_path)
            writers_episode.write(row)
            done.add(key)
            reused += 1
    jobs = [(seed, p, method, label, model) for label, model in models.items()
            for p in PS for method in methods for seed in range(args.seed_start, args.seed_start + args.episodes)
            if (label, p, seed, method) not in done]
    random.Random(20261113).shuffle(jobs)
    expected = len(models) * len(PS) * len(methods) * args.episodes
    print(f"Run {run_id}: {len(jobs)} episodes remaining / {expected} "
          f"({reused} reused from {args.reuse_from}); concurrency={args.concurrency}", flush=True)
    writers = {name: JsonlWriter(path) for name, path in (("episodes", episodes_path), ("traces", traces_path),
                                                           ("raw", raw_path), ("failures", failures_path))}
    client = MagpieClient(args.base_url, args.api_key, args.timeout, args.request_retries, args.max_tokens)
    counts = Counter()
    start = time.monotonic()

    def execute(job):
        summary, trace, raw = run_episode(*job, client, args.parse_retries, run_id)
        if args.model_alias:
            # The route was swapped mid-run (e.g. a provider ran out of quota);
            # keep the results under the same model label so the statistics
            # treat them as one model, and record the actual route.
            for record in [summary, raw] + trace:
                if isinstance(record, dict):
                    record["actual_model"] = record.get("model")
                    record["model_label"] = args.model_alias
        return summary, trace, raw

    def record(summary, trace, raw) -> None:
        writers["episodes"].write(summary)
        for row in trace:
            writers["traces"].write(row)
        writers["raw"].write(raw)
        if summary["infrastructure_failure"]:
            writers["failures"].write(summary["infrastructure_failure_detail"])
        counts[(summary["model_label"], summary["disturbance_p"], summary["method"])] += 1
        with print_lock:
            print(f"[{counts.total()}/{len(jobs)}] {summary['model_label']} p={summary['disturbance_p']} "
                  f"seed={summary['seed']} {summary['method']}: success={summary['success']} "
                  f"reason={summary['failure_reason']} cost={summary['total_cost']} steps={summary['steps']} "
                  f"elapsed={time.monotonic()-start:.0f}s", flush=True)

    # Upstream capacity is per provider account, not per model: two models on the
    # same account share its queue, and CodeArts documents a hard 3-concurrent-
    # session cap that Magpie turns into soft queueing (which then trips our
    # wall-clock deadline). So the cap is per provider, not per model, and the
    # total concurrency is bounded by providers x cap.
    provider_jobs: dict[str, list] = {}
    for job in jobs:
        provider_jobs.setdefault(provider_of(job[4]), []).append(job)
    per_provider = max(1, min(args.provider_concurrency,
                              args.concurrency // max(1, len(provider_jobs))))
    print(f"Providers: { {k: len(v) for k, v in provider_jobs.items()} } "
          f"-> {per_provider} concurrent each (total {per_provider * len(provider_jobs)})", flush=True)

    futures = []
    pools = []
    for provider, provider_job_list in provider_jobs.items():
        pool = ThreadPoolExecutor(max_workers=per_provider, thread_name_prefix=provider)
        pools.append(pool)
        for job in provider_job_list:
            futures.append(pool.submit(execute, job))
    try:
        for future in as_completed(futures):
            try:
                summary, trace, raw = future.result()
            except Exception as exc:
                # A programming error is loud and does not silently become a policy failure.
                print(f"HARNESS ERROR: {type(exc).__name__}: {exc}\n{traceback.format_exc()}", flush=True)
                for pending in futures:
                    pending.cancel()
                raise
            record(summary, trace, raw)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    print(f"Completed in {time.monotonic()-start:.1f}s. Output: {output}", flush=True)

    # Repair pass: an upstream timeout is an infrastructure fault, not a policy
    # failure. Re-run those episodes with low concurrency so the comparison is
    # not contaminated by provider flakiness. The retry is recorded in the log.
    if args.repair_infrastructure:
        failed = [row for row in load_jsonl(episodes_path)
                  if row.get("infrastructure_failure")
                  and (row["model"], row["disturbance_p"], row["seed"], row["method"]) not in done]
        if failed:
            print(f"Repair pass: re-running {len(failed)} infrastructure-failed episodes "
                  f"at concurrency {args.repair_concurrency}", flush=True)
            repair_jobs = [(row["seed"], row["disturbance_p"], row["method"],
                            row["model_label"], row["model"]) for row in failed]
            with ThreadPoolExecutor(max_workers=args.repair_concurrency) as pool:
                for future in as_completed([pool.submit(execute, job) for job in repair_jobs]):
                    summary, trace, raw = future.result()
                    summary["repair_of_infrastructure_failure"] = True
                    record(summary, trace, raw)
            print(f"Repair pass finished. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
