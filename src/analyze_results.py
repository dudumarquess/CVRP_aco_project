from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ACS CVRP experiment results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory containing per-seed JSON files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to write summary CSV (default: <results_dir>/summary.csv).",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="Path to write summary Markdown (default: <results_dir>/summary.md).",
    )
    parser.add_argument(
        "--plot_instance",
        type=str,
        default=None,
        help="Instance name to plot convergence (optional).",
    )
    parser.add_argument(
        "--plot_out",
        type=str,
        default=None,
        help="Path for convergence plot (default: <results_dir>/convergence.png).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of variants to include (filters others out).",
    )
    return parser.parse_args()


def normalize_instance(name: str) -> str:
    """Strip directory/extension to compare instance names flexibly."""
    return Path(name).stem if name else ""


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_runs(results_dir: Path) -> List[Dict]:
    runs = []
    for path in results_dir.rglob("seed_*.json"):
        try:
            data = json.loads(path.read_text())
            runs.append(data)
        except Exception:
            continue
    return runs


def extract_best_cost(run: Dict) -> float | None:
    if "best_cost" in run:
        return float(run["best_cost"])
    final = run.get("final")
    if isinstance(final, dict) and "best_cost" in final:
        return float(final["best_cost"])
    return None


def extract_elapsed(run: Dict) -> float:
    if "elapsed_seconds" in run:
        return float(run["elapsed_seconds"])
    return float(run.get("runtime_seconds", 0.0))


def extract_best_history(run: Dict) -> List[float]:
    if "best_history" in run:
        return run["best_history"] or []
    final = run.get("final")
    if isinstance(final, dict):
        return final.get("best_history", []) or []
    return []


def aggregate(runs: List[Dict], allowed_variants: List[str] | None) -> Dict[Tuple[str, str], Dict]:
    grouped: Dict[Tuple[str, str], Dict] = {}
    for run in runs:
        instance_raw = run.get("instance_name") or run.get("instance") or run.get("problem_file")
        instance = normalize_instance(instance_raw)
        # allow legacy "baseline" but keep distinct
        variant = run.get("variant") or run.get("algorithm") or "unknown"
        if allowed_variants is not None and variant not in allowed_variants:
            continue
        best_cost = extract_best_cost(run)
        if not instance or not variant or best_cost is None:
            continue
        key = (instance, variant)
        grouped.setdefault(key, {"best_cost": [], "elapsed": [], "gap": [], "count": 0})
        grouped[key]["best_cost"].append(best_cost)
        grouped[key]["elapsed"].append(extract_elapsed(run))
        grouped[key]["count"] += 1
        gap = run.get("gap_percent")
        if gap is not None:
            grouped[key]["gap"].append(float(gap))
    return grouped


def write_csv(summary: Dict[Tuple[str, str], Dict], path: Path) -> None:
    fieldnames = [
        "instance",
        "variant",
        "n_runs",
        "best_cost_mean",
        "best_cost_std",
        "best_cost_median",
        "best_cost_min",
        "best_cost_max",
        "elapsed_mean",
        "elapsed_std",
        "elapsed_median",
        "elapsed_min",
        "elapsed_max",
        "gap_mean",
        "gap_median",
        "gap_n",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (instance, variant), metrics in summary.items():
            best_stats = compute_stats(metrics["best_cost"])
            elapsed_stats = compute_stats(metrics["elapsed"])
            row = {
                "instance": instance,
                "variant": variant,
                "n_runs": metrics.get("count", len(metrics["best_cost"])),
                "best_cost_mean": best_stats["mean"],
                "best_cost_std": best_stats["std"],
                "best_cost_median": best_stats["median"],
                "best_cost_min": best_stats["min"],
                "best_cost_max": best_stats["max"],
                "elapsed_mean": elapsed_stats["mean"],
                "elapsed_std": elapsed_stats["std"],
                "elapsed_median": elapsed_stats["median"],
                "elapsed_min": elapsed_stats["min"],
                "elapsed_max": elapsed_stats["max"],
                "gap_mean": "",
                "gap_median": "",
                "gap_n": 0,
            }
            if metrics["gap"]:
                gap_stats = compute_stats(metrics["gap"])
                row["gap_mean"] = gap_stats["mean"]
                row["gap_median"] = gap_stats["median"]
                row["gap_n"] = len(metrics["gap"])
            writer.writerow(row)


def write_md(summary: Dict[Tuple[str, str], Dict], path: Path) -> None:
    lines = []
    lines.append("| instance | variant | n | best_mean | best_std | best_median | best_min | best_max | time_mean | time_std | gap_mean | gap_median |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (instance, variant), metrics in summary.items():
        best_stats = compute_stats(metrics["best_cost"])
        elapsed_stats = compute_stats(metrics["elapsed"])
        gap_mean = gap_median = "-"
        if metrics["gap"]:
            gap_stats = compute_stats(metrics["gap"])
            gap_mean = f"{gap_stats['mean']:.2f}"
            gap_median = f"{gap_stats['median']:.2f}"
        lines.append(
            "| {i} | {v} | {n} | {bm:.2f} | {bs:.2f} | {bmed:.2f} | {bmin:.2f} | {bmax:.2f} | {tm:.2f} | {ts:.2f} | {gmean} | {gmed} |".format(
                i=instance,
                v=variant,
                n=metrics.get("count", len(metrics["best_cost"])),
                bm=best_stats["mean"],
                bs=best_stats["std"],
                bmed=best_stats["median"],
                bmin=best_stats["min"],
                bmax=best_stats["max"],
                tm=elapsed_stats["mean"],
                ts=elapsed_stats["std"],
                gmean=gap_mean,
                gmed=gap_median,
            )
        )
    path.write_text("\n".join(lines))


def plot_convergence(summary_runs: List[Dict], instance: str, out_path: Path, allowed_variants: List[str] | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping convergence plot.")
        return

    target = normalize_instance(instance)
    by_variant: Dict[str, List[List[float]]] = {}
    for run in summary_runs:
        inst = normalize_instance(run.get("instance_name") or run.get("instance") or run.get("problem_file"))
        if inst != target:
            continue
        variant = run.get("variant") or run.get("algorithm") or "unknown"
        if allowed_variants is not None and variant not in allowed_variants:
            continue
        if variant is None:
            continue
        by_variant.setdefault(variant, []).append(extract_best_history(run))

    if not by_variant:
        print(f"No runs found for instance {instance}; skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    for variant, histories in by_variant.items():
        if not histories:
            continue
        min_len = min(len(h) for h in histories)
        trimmed = np.array([h[:min_len] for h in histories], dtype=float)
        mean_curve = np.mean(trimmed, axis=0)
        plt.plot(mean_curve, label=variant)

    plt.xlabel("Iteration")
    plt.ylabel("Global best cost (mean)")
    plt.title(f"Convergence - {instance}")
    plt.grid(True)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Convergence plot saved to: {out_path}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "summary.csv"
    output_md = Path(args.output_md) if args.output_md else results_dir / "summary.md"
    plot_out = Path(args.plot_out) if args.plot_out else results_dir / "convergence.png"

    runs = load_runs(results_dir)
    if not runs:
        print("No runs found.")
        return

    grouped = aggregate(runs, args.variants)
    write_csv(grouped, output_csv)
    write_md(grouped, output_md)

    print(f"Summary CSV written to: {output_csv}")
    print(f"Summary MD written to: {output_md}")

    if args.plot_instance:
        plot_convergence(runs, args.plot_instance, plot_out, args.variants)


if __name__ == "__main__":
    main()
