from pathlib import Path

from .problem import CVRPProblem
from .aco import AntColony, ACSParams


def main():
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

    acs = AntColony(problem, params)

    best = acs.run()

    print("\n=== Final Result ===")
    print("Best Cost:", best.total_cost(problem))
    print("Number of routes:", len(best.routes))
    for idx, r in enumerate(best.routes, start=1):
        print(f"Route {idx}: {r.nodes}")


if __name__ == "__main__":
    main()
