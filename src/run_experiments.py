from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .aco import AntColony, ACSParams
from .problem import CVRPProblem


class IterationCollector:
    """Observer that stores per-iteration metrics."""

    def __init__(self) -> None:
        self.history: List[Dict[str, float]] = []

    def __call__(self, event: Dict[str, float]) -> None:
        if not isinstance(event, dict):
            return
        if event.get("type") != "iteration_end":
            return
        self.history.append(
            {
                "iteration": int(event["iteration"]),
                "best_iteration_cost": float(event["best_iteration_cost"]),
                "global_best_cost": float(event["global_best_cost"]),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ACS experiments for CVRP.")
    parser.add_argument(
        "--instance",
        type=str,
        default="data/A-n32-k5.vrp",
        help="VRPLIB instance path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to store per-seed logs and summaries.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit list of seeds to run (overrides --n_seeds).",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=20,
        help="Number of seeds to run when --seeds is not provided.",
    )
    return parser.parse_args()


def baseline_params() -> ACSParams:
    return ACSParams(
        n_ants=10,
        n_iterations=30,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        xi=0.1,
        q0=0.9,
        tau0=0.01,
    )


def compute_best_iter(history: Sequence[Dict[str, float]], total_iters: int) -> Tuple[int, int]:
    best_iter = 1
    best_cost = None
    for entry in history:
        cost = entry["global_best_cost"]
        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
            best_iter = entry["iteration"]
    stagnation = max(0, total_iters - best_iter)
    return best_iter, stagnation


def run_single(
    problem: CVRPProblem,
    params: ACSParams,
    seed: int,
    instance_name: str,
    out_dir: Path,
) -> Dict[str, float]:
    collector = IterationCollector()
    acs = AntColony(problem, params, observer=collector, seed=seed)

    start = time.perf_counter()
    best = acs.run()
    runtime_seconds = time.perf_counter() - start

    best_cost = float(best.total_cost(problem))
    routes = [list(map(int, route.nodes)) for route in best.routes]
    n_routes = len(routes)

    best_iter_found, stagnation_iters = compute_best_iter(collector.history, params.n_iterations)

    record = {
        "instance": instance_name,
        "seed": seed,
        "params": vars(params),
        "runtime_seconds": runtime_seconds,
        "history": collector.history,
        "final": {
            "best_cost": best_cost,
            "n_routes": n_routes,
            "routes": routes,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"seed_{seed:03d}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return {
        "seed": seed,
        "best_cost": best_cost,
        "n_routes": n_routes,
        "runtime_seconds": runtime_seconds,
        "best_iter_found": best_iter_found,
        "stagnation_iters": stagnation_iters,
    }


def write_summary_csv(rows: List[Dict[str, float]], path: Path) -> None:
    fieldnames = [
        "seed",
        "best_cost",
        "n_routes",
        "runtime_seconds",
        "best_iter_found",
        "stagnation_iters",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_stats(costs: List[float], path: Path) -> None:
    arr = np.array(costs, dtype=float)
    stats = {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent

    instance_path = Path(args.instance)
    if not instance_path.is_absolute():
        instance_path = base_dir / instance_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    problem = CVRPProblem.from_vrplib(instance_path)
    params = baseline_params()
    instance_name = instance_path.stem

    summary_rows: List[Dict[str, float]] = []
    best_costs: List[float] = []

    for seed in seeds:
        row = run_single(problem, params, seed, instance_name, out_dir)
        summary_rows.append(row)
        best_costs.append(row["best_cost"])

    write_summary_csv(summary_rows, out_dir / "summary.csv")
    write_summary_stats(best_costs, out_dir / "summary.json")

    best_row = min(summary_rows, key=lambda r: r["best_cost"])
    mean_cost = float(np.mean(best_costs))
    std_cost = float(np.std(best_costs))

    print(f"Runs completed: {len(seeds)}")
    print(f"Best seed: {best_row['seed']} with cost {best_row['best_cost']:.2f}")
    print(f"Mean best cost: {mean_cost:.2f} ± {std_cost:.2f}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
