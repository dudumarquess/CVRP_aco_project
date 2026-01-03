from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from .aco import ACSParams, AntColony
from .experiment_logger import save_experiment
from .problem import CVRPProblem
from .viewer import ACSLiveViewer

BASE_DIR = Path(__file__).resolve().parent.parent

BEST_KNOWN_COSTS = {
    "A-n32-k5": 784.0,
    "A-n80-k10": 1763.0,
}

MAX_PHEROMONE_POINTS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACS smoketest")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the ACS RNG (default: 42)",
    )
    parser.add_argument(
        "--problem-file",
        type=Path,
        default=BASE_DIR / "data" / "A-n32-k5.vrp",
        help="Path to a VRPLIB instance file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ACS",
        help='Algorithm name for logging (default: "ACS").',
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        help='Variant tag for logging (e.g., "baseline", "ls_fixed", "ls_adaptive").',
    )
    return parser.parse_args()


def compute_problem_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lookup_best_known(instance_name: str):
    return BEST_KNOWN_COSTS.get(instance_name)


def compute_gap_percent(best_cost: float, best_known: float | None) -> float | None:
    if best_known is None or best_known == 0:
        return None
    return (best_cost - best_known) / best_known * 100.0


def relative_problem_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=BASE_DIR,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def feasibility_report(solution, problem) -> dict:
    counts = Counter()
    capacity_violations = []
    for idx, route in enumerate(solution.routes):
        load = route.load(problem)
        if load > problem.capacity:
            capacity_violations.append(
                {"route_index": idx, "load": load, "capacity": problem.capacity}
            )
        counts.update(n for n in route.nodes if n != 0)

    expected = set(range(1, problem.n_customers + 1))
    visited = set(counts.keys())
    unvisited_customers = sorted(expected - visited)
    duplicate_customers = sorted([n for n, c in counts.items() if c > 1])

    feasible = (
        not capacity_violations
        and not unvisited_customers
        and not duplicate_customers
    )

    return {
        "feasible": feasible,
        "capacity_violations": len(capacity_violations),
        "capacity_violation_routes": capacity_violations,
        "unvisited_customers": unvisited_customers,
        "duplicate_customers": duplicate_customers,
    }


def main():
    args = parse_args()
    path = Path(args.problem_file)
    if not path.is_absolute():
        path = (BASE_DIR / args.problem_file).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Problem file not found: {path}")

    problem = CVRPProblem.from_vrplib(path)

    print("Number of customers:", problem.n_customers)
    print("Capacity:", problem.capacity)

    params = ACSParams(
        n_ants=10,
        n_iterations=30,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        xi=0.1,
        q0=0.9,
        tau0=0.01,
    )

    # 👇 1) cria o viewer
    viewer = ACSLiveViewer(problem, pause=0.1)

    # 👇 2) passa o viewer como observer
    acs = AntColony(problem, params, observer=viewer, seed=args.seed)

    start_time = time.perf_counter()
    best = acs.run()
    elapsed = time.perf_counter() - start_time
    best_cost = best.total_cost(problem)
    instance_name = path.stem
    best_known = lookup_best_known(instance_name)
    gap_percent = compute_gap_percent(best_cost, best_known)
    git_commit = get_git_commit()
    problem_sha1 = compute_problem_sha1(path)

    pheromone_history = acs.pheromone_history
    if len(pheromone_history) > MAX_PHEROMONE_POINTS:
        stride = max(1, len(pheromone_history) // MAX_PHEROMONE_POINTS)
        pheromone_history = pheromone_history[::stride]
        if pheromone_history[-1]["iteration"] != acs.params.n_iterations:
            pheromone_history.append(acs.pheromone_history[-1])

    print("\n=== Final Result ===")
    print("Best Cost:", best_cost)
    print("Number of routes:", len(best.routes))
    for idx, r in enumerate(best.routes, start=1):
        print(f"Route {idx}: {r.nodes}")

    feasibility = feasibility_report(best, problem)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "instance_name": instance_name,
        "problem_file": relative_problem_path(path),
        "problem_sha1": problem_sha1,
        "n_customers": problem.n_customers,
        "capacity": problem.capacity,
        "seed": args.seed,
        "algorithm": args.algorithm,
        "variant": args.variant,
        "params": vars(params),
        "termination_criteria": {
            "type": "fixed_iterations",
            "n_iterations": params.n_iterations,
        },
        "best_known": best_known,
        "best_cost": best_cost,
        "gap_percent": gap_percent,
        "n_routes": len(best.routes),
        "routes": [route.nodes for route in best.routes],
        "best_history": acs.best_history,
        "pheromone_history": pheromone_history,
        "elapsed_seconds": elapsed,
        "feasible": feasibility["feasible"],
        "capacity_violations": feasibility["capacity_violations"],
        "capacity_violation_routes": feasibility["capacity_violation_routes"],
        "unvisited_customers": feasibility["unvisited_customers"],
        "duplicate_customers": feasibility["duplicate_customers"],
        "python_version": platform.python_version(),
        "git_commit": git_commit,
    }
    experiments_dir = BASE_DIR / "experiments"
    saved_path = save_experiment(record, experiments_dir)
    print(f"\nExperiment saved to: {saved_path}")

    # 👇 3) mantém a janela aberta no final
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
