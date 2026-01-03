from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aco import ACSParams, AntColony
from .problem import CVRPProblem
from .solution import Solution, two_opt_solution


BEST_KNOWN_COSTS = {
    "A-n32-k5": 784.0,
    "A-n80-k10": 1763.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ACS experiments for CVRP.")
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        default=["data/A-n32-k5.vrp"],
        help="List of VRPLIB instance paths.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["acs_baseline", "acs_cl15", "acs_cl15_ls2opt_fixed", "acs_cl15_ls2opt_adaptive"],
        help="Variant labels to run.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0:19",
        help='Seed list "0,1,2" or range "0:19" (inclusive).',
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to store results.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=30,
        help="Override ACS iterations.",
    )
    parser.add_argument(
        "--n_ants",
        type=int,
        default=10,
        help="Override ACS number of ants.",
    )
    parser.add_argument(
        "--stagnation_threshold",
        type=int,
        default=10,
        help="Adaptive LS: iterations without improvement before widening LS.",
    )
    parser.add_argument(
        "--adaptive_top_m",
        type=int,
        default=3,
        help="Adaptive LS: how many top solutions to improve when stagnant.",
    )
    return parser.parse_args()


def parse_seed_spec(spec: str) -> List[int]:
    if not spec:
        return list(range(20))
    if ":" in spec:
        start_s, end_s = spec.split(":")
        start = int(start_s) if start_s else 0
        end = int(end_s)
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def compute_problem_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_gap_percent(best_cost: float, best_known: float | None) -> float | None:
    if best_known is None or best_known == 0:
        return None
    return (best_cost - best_known) / best_known * 100.0


def get_git_commit(base_dir: Path) -> Optional[str]:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=base_dir,
        )
        return result.stdout.strip()
    except Exception:
        return None


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def configure_variant(
    label: str,
    base_params: ACSParams,
    args: argparse.Namespace,
) -> Tuple[ACSParams, str, Optional[callable], int, int, int]:
    """Returns params, ls_mode, ls_fn, adaptive_threshold, adaptive_top_m, expected_k."""
    params = replace(base_params)
    ls_mode = "none"
    ls_fn = None
    threshold = args.stagnation_threshold
    top_m = args.adaptive_top_m
    expected_k = params.candidate_list_size

    if label == "acs_baseline":
        params.candidate_list_size = 0
        expected_k = 0
    elif label == "acs_cl15":
        params.candidate_list_size = 15
        expected_k = 15
    elif label == "acs_cl15_ls2opt_fixed":
        params.candidate_list_size = 15
        ls_mode = "best"
        ls_fn = two_opt_solution
        expected_k = 15
    elif label == "acs_cl15_ls2opt_adaptive":
        params.candidate_list_size = 15
        ls_mode = "adaptive"
        ls_fn = two_opt_solution
        expected_k = 15
    else:
        raise ValueError(f"Unknown variant: {label}")

    return params, ls_mode, ls_fn, threshold, top_m, expected_k


def run_single(
    problem: CVRPProblem,
    params: ACSParams,
    seed: int,
    instance_name: str,
    variant: str,
    out_dir: Path,
    problem_file: str,
    problem_sha1: str,
    best_known: float | None,
    git_commit: Optional[str],
    ls_mode: str,
    ls_fn,
    adaptive_threshold: int,
    adaptive_top_m: int,
) -> Dict[str, object]:
    assert params.candidate_list_size == params.candidate_list_size, "Params mutated unexpectedly."

    acs = AntColony(
        problem,
        params,
        observer=None,
        seed=seed,
        local_search_fn=ls_fn,
        local_search_mode=ls_mode,
        adaptive_threshold=adaptive_threshold,
        adaptive_top_m=adaptive_top_m,
    )

    start = time.perf_counter()
    best = acs.run()
    elapsed = time.perf_counter() - start

    best_cost = float(best.total_cost(problem))
    n_routes = len(best.routes)
    routes = [route.nodes for route in best.routes]
    feasible = best.is_feasible(problem)

    gap_percent = compute_gap_percent(best_cost, best_known)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "instance_name": instance_name,
        "problem_file": problem_file,
        "problem_sha1": problem_sha1,
        "seed": seed,
        "algorithm": "ACS",
        "variant": variant,
        "params": vars(params),
        "best_known": best_known,
        "best_cost": best_cost,
        "gap_percent": gap_percent,
        "n_routes": n_routes,
        "routes": routes,
        "best_history": acs.best_history,
        "pheromone_history": acs.pheromone_history,
        "elapsed_seconds": elapsed,
        "feasible": feasible,
        "git_commit": git_commit,
        "ls_mode": ls_mode,
        "adaptive_threshold": adaptive_threshold,
        "adaptive_top_m": adaptive_top_m,
        "candidate_list_size": params.candidate_list_size,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"seed_{seed}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return {
        "best_cost": best_cost,
        "elapsed_seconds": elapsed,
        "gap_percent": gap_percent,
        "feasible": feasible,
        "path": str(json_path),
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    seeds = parse_seed_spec(args.seeds)
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = base_dir / out_root

    git_commit = get_git_commit(base_dir)

    for instance in args.instances:
        instance_path = Path(instance)
        if not instance_path.is_absolute():
            instance_path = base_dir / instance_path
        if not instance_path.exists():
            raise FileNotFoundError(f"Instance not found: {instance_path}")

        problem = CVRPProblem.from_vrplib(instance_path)
        instance_name = instance_path.stem
        problem_sha1 = compute_problem_sha1(instance_path)
        problem_file = relative_path(instance_path, base_dir)
        best_known = BEST_KNOWN_COSTS.get(instance_name)

        print(f"\n=== Instance: {instance_name} ({problem_file}) ===")

        base_params = ACSParams(
            n_ants=args.n_ants,
            n_iterations=args.n_iterations,
        )

        for variant in args.variants:
            params, ls_mode, ls_fn, thr, top_m, expected_k = configure_variant(variant, base_params, args)
            assert params.candidate_list_size == expected_k, (
                f"Variant {variant} expected candidate_list_size {expected_k}, "
                f"got {params.candidate_list_size}"
            )
            print(f"- Variant: {variant} | seeds: {seeds}")

            variant_dir = out_root / instance_name / variant

            for seed in seeds:
                result = run_single(
                    problem=problem,
                    params=params,
                    seed=seed,
                    instance_name=instance_name,
                    variant=variant,
                    out_dir=variant_dir,
                    problem_file=problem_file,
                    problem_sha1=problem_sha1,
                    best_known=best_known,
                    git_commit=git_commit,
                    ls_mode=ls_mode,
                    ls_fn=ls_fn,
                    adaptive_threshold=thr,
                    adaptive_top_m=top_m,
                )
                print(
                    f"  seed {seed:>3} -> cost {result['best_cost']:.2f}, "
                    f"time {result['elapsed_seconds']:.2f}s, "
                    f"feasible={result['feasible']}, "
                    f"cl_size={params.candidate_list_size}"
                )


if __name__ == "__main__":
    main()
