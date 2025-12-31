import argparse
import time
from datetime import datetime
from pathlib import Path

from .problem import CVRPProblem
from .aco import AntColony, ACSParams
from .viewer import ACSLiveViewer
from .experiment_logger import save_experiment

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACS smoketest")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the ACS RNG (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = BASE_DIR / "data" / "A-n32-k5.vrp"

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

    print("\n=== Final Result ===")
    print("Best Cost:", best.total_cost(problem))
    print("Number of routes:", len(best.routes))
    for idx, r in enumerate(best.routes, start=1):
        print(f"Route {idx}: {r.nodes}")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem_file": str(path),
        "n_customers": problem.n_customers,
        "capacity": problem.capacity,
        "seed": args.seed,
        "params": vars(params),
        "best_cost": best.total_cost(problem),
        "n_routes": len(best.routes),
        "routes": [route.nodes for route in best.routes],
        "best_history": acs.best_history,
        "elapsed_seconds": elapsed,
    }
    experiments_dir = BASE_DIR / "experiments"
    saved_path = save_experiment(record, experiments_dir)
    print(f"\nExperiment saved to: {saved_path}")

    # 👇 3) mantém a janela aberta no final
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
