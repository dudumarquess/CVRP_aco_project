from __future__ import annotations

# Example:
# python -m src.run_experiments --instances_glob "data/*.vrp" \
#   --variants acs_nocl_2opt_q0sched acs_clsqrtn_2opt_q0sched \
#   --seeds 0:29 --n_iterations 30 --n_ants 10 --out_dir results

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aco import ACSParams, AntColony
from .problem import CVRPProblem
from .solution import Solution, two_opt_solution


BEST_KNOWN_COSTS = {
    "A-n32-k5": 784.0,
    "A-n33-k5": 661.0,
    "A-n33-k6": 742.0,
    "A-n34-k5": 778.0,
    "A-n36-k5": 799.0,
    "A-n37-k5": 669.0,
    "A-n37-k6": 949.0,
    "A-n38-k5": 730.0,
    "A-n39-k5": 822.0,
    "A-n39-k6": 831.0,
    "A-n44-k6": 937.0,
    "A-n45-k6": 944.0,
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
        "--instances_glob",
        type=str,
        nargs="*",
        default=None,
        help='Glob(s) of instance paths, relative to repo root (e.g., "data/A/*.vrp").',
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=[
            "acs_baseline",
            "acs_cl15",
            "acs_cl15_ls2opt_fixed",
            "acs_cl15_ls2opt_fixed_q0sched",
            "acs_nocl_2opt_q0sched",
            "acs_clsqrtn_2opt_q0sched",
        ],
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
) -> Tuple[ACSParams, str, Optional[callable], int]:
    """Returns params, ls_mode, ls_fn, expected_k."""
    params = replace(base_params)
    ls_mode = "none"
    ls_fn = None
    expected_k = params.candidate_list_size

    if label == "acs_baseline":
        params.candidate_list_mode = "none"
        params.candidate_list_size = 0
        expected_k = 0
    elif label == "acs_cl15":
        params.candidate_list_mode = "fixed"
        params.candidate_list_size = 15
        expected_k = 15
    elif label == "acs_cl15_ls2opt_fixed":
        params.candidate_list_mode = "fixed"
        params.candidate_list_size = 15
        ls_mode = "best"
        ls_fn = two_opt_solution
        expected_k = 15
    elif label == "acs_cl15_ls2opt_fixed_q0sched":
        params.candidate_list_mode = "fixed"
        params.candidate_list_size = 15
        ls_mode = "best"
        ls_fn = two_opt_solution
        params.use_q0_schedule = True
        expected_k = 15
    elif label == "acs_nocl_q0sched":
        params.candidate_list_mode = "none"
        params.candidate_list_size = 0
        params.use_q0_schedule = True
        expected_k = 0
    elif label == "acs_nocl_2opt":
        params.candidate_list_mode = "none"
        params.candidate_list_size = 0
        ls_mode = "best"
        ls_fn = two_opt_solution
        expected_k = 0
    elif label == "acs_nocl_2opt_q0sched":
        params.candidate_list_mode = "none"
        params.candidate_list_size = 0
        ls_mode = "best"
        ls_fn = two_opt_solution
        params.use_q0_schedule = True
        expected_k = 0
    elif label == "acs_clsqrtn_none":
        params.candidate_list_mode = "sqrt"
    elif label == "acs_clsqrtn_q0sched":
        params.candidate_list_mode = "sqrt"
        params.use_q0_schedule = True
    elif label == "acs_clsqrtn_2opt":
        params.candidate_list_mode = "sqrt"
        ls_mode = "best"
        ls_fn = two_opt_solution
    elif label == "acs_clsqrtn_2opt_q0sched":
        params.candidate_list_mode = "sqrt"
        ls_mode = "best"
        ls_fn = two_opt_solution
        params.use_q0_schedule = True
    else:
        raise ValueError(f"Unknown variant: {label}")

    if "q0" in label:
        assert params.use_q0_schedule, f"Variant {label} should enable q0 schedule"
    if "nocl" in label:
        assert params.candidate_list_mode == "none", f"Variant {label} should disable candidate list"
    if "clsqrtn" in label:
        assert params.candidate_list_mode == "sqrt", f"Variant {label} should use sqrt candidate list"

    return params, ls_mode, ls_fn, expected_k


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
) -> Dict[str, object]:
    assert params.candidate_list_size == params.candidate_list_size, "Params mutated unexpectedly."

    acs = AntColony(
        problem,
        params,
        observer=None,
        seed=seed,
        local_search_fn=ls_fn,
        local_search_mode=ls_mode,
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
        "candidate_list_size": params.candidate_list_size,
        "candidate_list_mode": params.candidate_list_mode,
        "candidate_list_k_effective": acs.candidate_list_k_effective,
        "ls_stats": getattr(acs, "ls_stats", {}),
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

    instance_paths: List[Path] = []
    if args.instances:
        instance_paths.extend(Path(p) for p in args.instances)
    if args.instances_glob:
        for pattern in args.instances_glob:
            full_pattern = str(base_dir / pattern) if not Path(pattern).is_absolute() else pattern
            instance_paths.extend(Path(p) for p in glob(full_pattern))

    if not instance_paths:
        instance_paths = [base_dir / "data" / "A-n32-k5.vrp"]

    resolved = {}
    for p in instance_paths:
        path = p if p.is_absolute() else (base_dir / p)
        resolved[str(path.resolve())] = path.resolve()
    instance_paths = [resolved[k] for k in sorted(resolved.keys())]

    git_commit = get_git_commit(base_dir)

    for instance_path in instance_paths:
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
            params, ls_mode, ls_fn, expected_k = configure_variant(variant, base_params, args)
            assert params.candidate_list_size == expected_k, (
                f"Variant {variant} expected candidate_list_size {expected_k}, "
                f"got {params.candidate_list_size}"
            )
            k_effective = 0
            mode = (params.candidate_list_mode or "fixed").lower()
            if mode == "none":
                k_effective = 0
            elif mode == "fixed":
                k_effective = max(0, int(params.candidate_list_size))
            elif mode == "sqrt":
                k_effective = int(math.ceil(math.sqrt(max(1, problem.n_customers))))
                k_effective = max(1, k_effective)
            print(
                f"- Variant: {variant} | seeds: {seeds} | mode={mode} "
                f"| k_effective={k_effective} | ls_mode={ls_mode} "
                f"| q0_schedule={params.use_q0_schedule}"
            )

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
                )
                print(
                    f"  seed {seed:>3} -> cost {result['best_cost']:.2f}, "
                    f"time {result['elapsed_seconds']:.2f}s, "
                    f"feasible={result['feasible']}, "
                    f"cl_size={params.candidate_list_size}"
                )


if __name__ == "__main__":
    main()
