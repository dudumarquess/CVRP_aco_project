from __future__ import annotations

import argparse
import csv
import json
import sys
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
        "--output_long_csv",
        type=str,
        default=None,
        help="Path to write per-run long CSV (default: <results_dir>/runs_long.csv).",
    )
    parser.add_argument(
        "--plot_instance",
        type=str,
        default=None,
        help="Instance name to plot convergence (optional).",
    )
    parser.add_argument(
        "--plot_instances",
        type=str,
        nargs="*",
        default=None,
        help="List of instance names to plot convergence for (optional).",
    )
    parser.add_argument(
        "--plot_all_instances",
        action="store_true",
        help="Plot convergence for all instances found.",
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
    parser.add_argument(
        "--aggregate_ls_stats",
        action="store_true",
        help="Include LS stats (widen_rate, ls_per_iter) in summaries when available.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if seed pairing mismatches are detected.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run Demsar-style statistical analysis (ranks + Friedman + post-hoc).",
    )
    parser.add_argument(
        "--score",
        type=str,
        choices=["median", "mean"],
        default="median",
        help="Aggregate per-variant score over seeds (median or mean best_cost).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests.",
    )
    parser.add_argument(
        "--control",
        type=str,
        default=None,
        help="Control variant for post-hoc tests (defaults to best average rank).",
    )
    return parser.parse_args()


def normalize_instance(name: str) -> str:
    """Strip directory/extension to compare instance names flexibly."""
    return Path(name).stem if name else ""


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0, "q25": 0.0, "q75": 0.0}
    ddof = 1 if arr.size > 1 else 0
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=ddof)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def load_runs(results_dir: Path) -> List[Dict]:
    runs = []
    warn_count = 0
    warn_cap = 20
    for path in results_dir.rglob("seed_*.json"):
        try:
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            runs.append(data)
        except Exception as e:
            warn_count += 1
            if warn_count <= warn_cap:
                print(f"[WARN] Failed to parse {path}: {e}")
            elif warn_count == warn_cap + 1:
                print("[WARN] Further JSON parse errors suppressed...")
            continue
    return runs


def load_best_known_from_sol(sol_dir: Path) -> Dict[str, float]:
    """Parse best-known costs from .sol files (VRPLIB A set)."""
    best = {}
    if not sol_dir.exists():
        return best
    for sol_file in sol_dir.glob("*.sol"):
        try:
            text = sol_file.read_text().strip().splitlines()
        except Exception:
            continue
        cost_val = None
        for line in reversed(text):
            if "cost" in line.lower():
                tokens = line.split()
                for tok in reversed(tokens):
                    try:
                        cost_val = float(tok)
                        break
                    except ValueError:
                        continue
                if cost_val is not None:
                    break
        if cost_val is not None:
            best[sol_file.stem] = cost_val
    return best


def enrich_runs_with_best_known(runs: List[Dict], best_known: Dict[str, float]) -> None:
    """Attach best_known/gap_percent when possible without breaking existing fields."""
    for run in runs:
        inst = normalize_instance(run.get("instance_name") or run.get("instance") or run.get("problem_file"))
        bk = best_known.get(inst)
        if bk is None:
            continue
        best_cost = extract_best_cost(run)
        if best_cost is None:
            continue
        run.setdefault("best_known", bk)
        gap = run.get("gap_percent")
        if gap is None and bk != 0:
            run["gap_percent"] = (best_cost - bk) / bk * 100.0


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
        grouped.setdefault(key, {"best_cost": [], "elapsed": [], "gap": [], "count": 0, "widen_rate": [], "ls_per_iter": []})
        grouped[key]["best_cost"].append(best_cost)
        grouped[key]["elapsed"].append(extract_elapsed(run))
        grouped[key]["count"] += 1
        gap = run.get("gap_percent")
        if gap is not None:
            grouped[key]["gap"].append(float(gap))
        ls_stats = run.get("ls_stats") or {}
        if "widen_rate" in ls_stats:
            grouped[key]["widen_rate"].append(float(ls_stats.get("widen_rate", 0.0)))
        if "ls_per_iter" in ls_stats:
            grouped[key]["ls_per_iter"].append(float(ls_stats.get("ls_per_iter", 0.0)))
    return grouped


def pairing_report(runs: List[Dict], allowed_variants: List[str] | None) -> Tuple[Dict[str, Dict[str, List[int]]], List[str]]:
    """Return seed sets per instance/variant and warnings (deterministic)."""
    seeds: Dict[str, Dict[str, List[int]]] = {}
    warnings: List[str] = []
    # collect seeds
    for run in runs:
        instance_raw = run.get("instance_name") or run.get("instance") or run.get("problem_file")
        instance = normalize_instance(instance_raw)
        variant = run.get("variant") or run.get("algorithm") or "unknown"
        if allowed_variants is not None and variant not in allowed_variants:
            continue
        if not instance or not variant:
            continue
        seed = run.get("seed")
        try:
            seed = int(seed)
        except Exception:
            seed = None
        if seed is None:
            continue
        seeds.setdefault(instance, {}).setdefault(variant, []).append(seed)

    # process per instance deterministically
    for inst in sorted(seeds.keys()):
        vmap = seeds[inst]
        # detect duplicates and dedupe
        for variant in sorted(vmap.keys()):
            original = vmap[variant]
            uniq = list(sorted(set(original)))
            if len(uniq) != len(original):
                from collections import Counter
                counts = Counter(original)
                dupes = sorted([s for s, c in counts.items() if c > 1])
                warnings.append(f"[WARN] {inst}: variant {variant} has duplicate seeds: {dupes}")
            vmap[variant] = uniq

        # choose reference set as mode of seed sets, tie-break by larger size then lexicographic
        from collections import Counter

        counter = Counter(frozenset(s) for s in vmap.values())
        ref_set = set()
        if counter:
            max_count = max(counter.values())
            candidates = [fs for fs, c in counter.items() if c == max_count]
            # tie-breaker: largest size
            max_size = max(len(fs) for fs in candidates)
            candidates = [fs for fs in candidates if len(fs) == max_size]
            # tie-breaker: lexicographically smallest sorted list
            candidates.sort(key=lambda fs: list(sorted(fs)))
            ref_set = set(candidates[0])

        for variant in sorted(vmap.keys()):
            seed_set = set(vmap[variant])
            missing = sorted(ref_set - seed_set)
            extra = sorted(seed_set - ref_set)
            if missing:
                warnings.append(f"[WARN] {inst}: variant {variant} missing seeds: {missing}")
            if extra:
                warnings.append(f"[WARN] {inst}: variant {variant} has extra seeds: {extra}")
    return seeds, warnings


def write_long_csv(runs: List[Dict], path: Path, allowed_variants: List[str] | None) -> None:
    fieldnames = [
        "instance",
        "variant",
        "seed",
        "best_cost",
        "gap_percent",
        "elapsed_seconds",
        "feasible",
        "git_commit",
        "n_routes",
        "candidate_list_size",
        "ls_mode",
        "adaptive_threshold",
        "adaptive_top_m",
        "total_ls_applications",
        "adaptive_widenings",
        "widen_rate",
        "ls_per_iter",
        "max_stagnation",
        "use_q0_schedule",
        "q0_min",
        "q0_max",
        "use_rho_adaptive",
        "rho",
        "rho_max",
        "best_known",
    ]
    rows = []
    for run in runs:
        instance_raw = run.get("instance_name") or run.get("instance") or run.get("problem_file")
        instance = normalize_instance(instance_raw)
        variant = run.get("variant") or run.get("algorithm") or "unknown"
        if allowed_variants is not None and variant not in allowed_variants:
            continue
        seed = run.get("seed")
        bc = extract_best_cost(run)
        try:
            seed = int(seed)
        except Exception:
            seed = seed if seed is not None else ""
        row = {
            "instance": instance,
            "variant": variant,
            "seed": seed,
            "best_cost": bc if bc is not None else "",
            "gap_percent": run.get("gap_percent", ""),
            "elapsed_seconds": run.get("elapsed_seconds", run.get("runtime_seconds", "")),
            "feasible": run.get("feasible", ""),
            "git_commit": run.get("git_commit", ""),
            "n_routes": run.get("n_routes", run.get("final", {}).get("n_routes") if isinstance(run.get("final"), dict) else ""),
            "candidate_list_size": run.get("candidate_list_size", run.get("params", {}).get("candidate_list_size") if isinstance(run.get("params"), dict) else ""),
            "ls_mode": run.get("ls_mode", ""),
            "adaptive_threshold": run.get("adaptive_threshold", ""),
            "adaptive_top_m": run.get("adaptive_top_m", ""),
            "total_ls_applications": "",
            "adaptive_widenings": "",
            "widen_rate": "",
            "ls_per_iter": "",
            "max_stagnation": "",
            "use_q0_schedule": "",
            "q0_min": "",
            "q0_max": "",
            "use_rho_adaptive": "",
            "rho": "",
            "rho_max": "",
            "best_known": "",
        }
        ls_stats = run.get("ls_stats") or {}
        for key in ["total_ls_applications", "adaptive_widenings", "widen_rate", "ls_per_iter", "max_stagnation"]:
            if key in ls_stats:
                row[key] = ls_stats.get(key, "")
        params = run.get("params") or {}
        for key in ["use_q0_schedule", "q0_min", "q0_max", "use_rho_adaptive", "rho", "rho_max"]:
            if key in params:
                row[key] = params.get(key, "")
        if "best_known" in run:
            row["best_known"] = run.get("best_known")
            if bc is not None and run["best_known"]:
                row["gap_percent"] = (bc - run["best_known"]) / run["best_known"] * 100.0
        rows.append(row)

    # sort for stability
    rows.sort(key=lambda r: (r["instance"], r["variant"], r["seed"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(summary: Dict[Tuple[str, str], Dict], path: Path, include_ls: bool) -> None:
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
    if include_ls:
        fieldnames.extend(["widen_rate_mean", "ls_per_iter_mean"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (instance, variant), metrics in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[0][1])):
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
            if include_ls:
                if metrics["widen_rate"]:
                    row["widen_rate_mean"] = compute_stats(metrics["widen_rate"])["mean"]
                else:
                    row["widen_rate_mean"] = ""
                if metrics["ls_per_iter"]:
                    row["ls_per_iter_mean"] = compute_stats(metrics["ls_per_iter"])["mean"]
                else:
                    row["ls_per_iter_mean"] = ""
            writer.writerow(row)


def write_md(summary: Dict[Tuple[str, str], Dict], path: Path, include_ls: bool) -> None:
    headers = ["instance", "variant", "n", "best_mean", "best_std", "best_median", "best_min", "best_max", "time_mean", "time_std", "gap_mean", "gap_median"]
    if include_ls:
        headers.extend(["widen_rate_mean", "ls_per_iter_mean"])
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for (instance, variant), metrics in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        best_stats = compute_stats(metrics["best_cost"])
        elapsed_stats = compute_stats(metrics["elapsed"])
        gap_mean = gap_median = "-"
        if metrics["gap"]:
            gap_stats = compute_stats(metrics["gap"])
            gap_mean = f"{gap_stats['mean']:.2f}"
            gap_median = f"{gap_stats['median']:.2f}"
        row = [
            instance,
            variant,
            str(metrics.get("count", len(metrics["best_cost"]))),
            f"{best_stats['mean']:.2f}",
            f"{best_stats['std']:.2f}",
            f"{best_stats['median']:.2f}",
            f"{best_stats['min']:.2f}",
            f"{best_stats['max']:.2f}",
            f"{elapsed_stats['mean']:.2f}",
            f"{elapsed_stats['std']:.2f}",
            gap_mean,
            gap_median,
        ]
        if include_ls:
            if metrics["widen_rate"]:
                row.append(f"{compute_stats(metrics['widen_rate'])['mean']:.4f}")
            else:
                row.append("-")
            if metrics["ls_per_iter"]:
                row.append(f"{compute_stats(metrics['ls_per_iter'])['mean']:.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
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
        hist = extract_best_history(run)
        if not hist:
            continue
        by_variant.setdefault(variant, []).append(hist)

    if not by_variant:
        print(f"No runs found for instance {instance}; skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    for variant in sorted(by_variant.keys()):
        histories = by_variant[variant]
        if not histories:
            continue
        min_len = min(len(h) for h in histories)
        max_len = max(len(h) for h in histories)
        if min_len != max_len:
            print(f"[INFO] Trimming histories for {instance}/{variant} to length {min_len} (max was {max_len})")
        if min_len == 0:
            continue
        trimmed = np.array([h[:min_len] for h in histories], dtype=float)
        mean_curve = np.mean(trimmed, axis=0)
        std_curve = np.std(trimmed, axis=0, ddof=1 if trimmed.shape[0] > 1 else 0)
        iters = np.arange(1, min_len + 1)
        plt.plot(iters, mean_curve, label=variant)
        if trimmed.shape[0] > 1:
            plt.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

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


def compute_instance_scores(
    runs: List[Dict],
    variants: List[str],
    score_mode: str,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, Dict[str, int]]]:
    """Compute per-instance scores (median/mean best_cost) per variant. Returns instances used, scores per variant aligned, and ignored counts."""
    variant_set = set(variants)
    # map instance -> variant -> seed -> best_cost (feasible only)
    inst_map: Dict[str, Dict[str, Dict[int, float]]] = {}
    ignored: Dict[str, Dict[str, int]] = {}
    for run in runs:
        inst = normalize_instance(run.get("instance_name") or run.get("instance") or run.get("problem_file"))
        var = run.get("variant") or run.get("algorithm") or "unknown"
        if variants and var not in variant_set:
            continue
        if not inst or not var:
            continue
        if run.get("feasible") is False:
            ignored.setdefault(inst, {}).setdefault(var, 0)
            ignored[inst][var] += 1
            continue
        seed = run.get("seed")
        try:
            seed = int(seed)
        except Exception:
            continue
        bc = extract_best_cost(run)
        if bc is None:
            continue
        inst_map.setdefault(inst, {}).setdefault(var, {})[seed] = bc

    # instances with all variants and matching seeds
    instances_used: List[str] = []
    scores: Dict[str, List[float]] = {v: [] for v in variants}
    for inst, vmap in sorted(inst_map.items()):
        if any(v not in vmap for v in variants):
            continue
        seed_sets = [set(vmap[v].keys()) for v in variants]
        if not all(s == seed_sets[0] for s in seed_sets):
            continue
        seeds = sorted(seed_sets[0])
        per_var_scores = {}
        for v in variants:
            vals = [vmap[v][s] for s in seeds if s in vmap[v]]
            if not vals:
                per_var_scores = None
                break
            if score_mode == "mean":
                per_var_scores[v] = float(np.mean(vals))
            else:
                per_var_scores[v] = float(np.median(vals))
        if per_var_scores is None:
            continue
        instances_used.append(inst)
        for v in variants:
            scores[v].append(per_var_scores[v])
    return instances_used, scores, ignored


def friedman_posthoc(
    instances: List[str],
    scores: Dict[str, List[float]],
    alpha: float,
    control: str | None,
) -> Dict[str, object]:
    """Run Friedman and post-hoc (Holm vs control)."""
    try:
        from scipy import stats
    except Exception as e:
        return {"error": f"scipy not available: {e}"}

    variants = list(scores.keys())
    if not instances or len(variants) < 2:
        return {"error": "Not enough data for statistical tests."}

    # prepare data matrix
    data = [scores[v] for v in variants]
    if not all(len(x) == len(instances) for x in data):
        return {"error": "Mismatched score lengths."}

    # Friedman
    try:
        fried = stats.friedmanchisquare(*data)
    except Exception as e:
        return {"error": f"Friedman failed: {e}"}

    # ranks per instance
    ranks = []
    for i in range(len(instances)):
        vals = [scores[v][i] for v in variants]
        r = stats.rankdata(vals, method="average")
        ranks.append(r)
    ranks = np.array(ranks)
    avg_ranks = {v: float(np.mean(ranks[:, idx])) for idx, v in enumerate(variants)}

    posthoc = {"method": None, "p_values": {}, "adjusted": {}, "control": None}
    if fried.pvalue < alpha and len(instances) >= 2:
        posthoc["method"] = "Holm vs control"
        # choose control
        ctrl = control
        if ctrl is None or ctrl not in variants:
            ctrl = min(avg_ranks.items(), key=lambda kv: kv[1])[0]
        posthoc["control"] = ctrl
        pvals = {}
        for v in variants:
            if v == ctrl:
                continue
            try:
                stat, p = stats.wilcoxon(
                    [scores[ctrl][i] for i in range(len(instances))],
                    [scores[v][i] for i in range(len(instances))],
                    zero_method="wilcox",
                )
            except Exception:
                p = 1.0
            pvals[v] = p
        # Holm correction
        m = len(pvals)
        ordered = sorted(pvals.items(), key=lambda kv: kv[1])
        adj = {}
        max_adj = 0.0
        for idx, (v, p) in enumerate(ordered):
            adj_p = (m - idx) * p
            adj_p = min(1.0, adj_p)
            max_adj = max(max_adj, adj_p)
            adj[v] = max_adj
        posthoc["p_values"] = pvals
        posthoc["adjusted"] = adj

    return {
        "friedman": {"stat": float(fried.statistic), "p_value": float(fried.pvalue)},
        "avg_ranks": avg_ranks,
        "variants": variants,
        "instances": instances,
        "posthoc": posthoc,
    }


def write_stats_report(stats_res: Dict[str, object], path: Path, alpha: float) -> None:
    lines = []
    if "error" in stats_res:
        lines.append(f"Stats error: {stats_res['error']}")
        path.write_text("\n".join(lines))
        return
    lines.append("## Statistical Analysis (Demsar-style)")
    lines.append("")
    lines.append(f"Instances used: {len(stats_res['instances'])}")
    lines.append(f"Variants: {', '.join(stats_res['variants'])}")
    lines.append("")
    lines.append("### Average Ranks (lower is better)")
    lines.append("| variant | avg_rank |")
    lines.append("|---|---|")
    for v, r in sorted(stats_res["avg_ranks"].items(), key=lambda kv: kv[1]):
        lines.append(f"| {v} | {r:.3f} |")
    lines.append("")
    fried = stats_res["friedman"]
    lines.append(f"Friedman chi^2 = {fried['stat']:.4f}, p = {fried['p_value']:.4g}")
    lines.append("")
    post = stats_res["posthoc"]
    if post.get("method"):
        lines.append(f"### Post-hoc ({post['method']}, alpha={alpha})")
        ctrl = post.get("control")
        if ctrl:
            lines.append(f"Control: **{ctrl}**")
        lines.append("| variant | p | p_holm | significant |")
        lines.append("|---|---|---|---|")
        for v, p in sorted(post.get("p_values", {}).items(), key=lambda kv: kv[1]):
            adj = post.get("adjusted", {}).get(v, p)
            sig = "yes" if adj < alpha else "no"
            lines.append(f"| {v} | {p:.4g} | {adj:.4g} | {sig} |")
    else:
        lines.append("No significant result or post-hoc not run.")
    path.write_text("\n".join(lines))


def plot_cd_diagram(stats_res: Dict[str, object], out_path: Path, alpha: float) -> None:
    if "error" in stats_res:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping CD diagram.")
        return
    variants = stats_res["variants"]
    if len(variants) < 2 or not stats_res["instances"]:
        return
    ranks = stats_res["avg_ranks"]
    # simple line plot of ranks with control highlight if any
    ctrl = stats_res.get("posthoc", {}).get("control")
    xs = [ranks[v] for v in variants]
    ys = [0] * len(variants)
    plt.figure(figsize=(12, 3))
    plt.hlines(0, min(xs) - 0.5, max(xs) + 0.5, colors="k", linewidth=1)
    for v, x in sorted(ranks.items(), key=lambda kv: kv[1]):
        color = "tab:red" if ctrl and v != ctrl and stats_res.get("posthoc", {}).get("adjusted", {}).get(v, 1.0) < alpha else "tab:blue"
        plt.plot(x, 0, "o", color=color)
        plt.text(x, 0.05, v, ha="center", va="bottom", rotation=45, fontsize=9)
    plt.xlabel("Average rank (lower is better)")
    plt.yticks([])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"CD diagram saved to: {out_path}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "summary.csv"
    output_md = Path(args.output_md) if args.output_md else results_dir / "summary.md"
    output_long = Path(args.output_long_csv) if args.output_long_csv else results_dir / "runs_long.csv"
    plot_out = Path(args.plot_out) if args.plot_out else results_dir / "convergence.png"
    base_dir = Path(__file__).resolve().parent.parent

    runs = load_runs(results_dir)
    if not runs:
        print("No runs found.")
        return

    # best-known from .sol (optional)
    best_known_map = load_best_known_from_sol(base_dir / "data" / "A")
    enrich_runs_with_best_known(runs, best_known_map)

    # pairing check
    seeds, warnings = pairing_report(runs, args.variants)
    for w in warnings:
        print(w)
    if args.strict and warnings:
        sys.exit(1)

    grouped = aggregate(runs, args.variants)
    write_csv(grouped, output_csv, include_ls=args.aggregate_ls_stats)
    write_md(grouped, output_md, include_ls=args.aggregate_ls_stats)
    write_long_csv(runs, output_long, args.variants)

    print(f"Summary CSV written to: {output_csv}")
    print(f"Summary MD written to: {output_md}")
    print(f"Runs long CSV written to: {output_long}")

    # statistical analysis
    if args.stats:
        variant_list = args.variants if args.variants else sorted({k[1] for k in grouped.keys()})
        instances_used, score_table, ignored = compute_instance_scores(runs, variant_list, args.score)
        if ignored:
            for inst, vmap in sorted(ignored.items()):
                for v, cnt in sorted(vmap.items()):
                    print(f"[INFO] Ignored {cnt} infeasible runs for {inst}/{v}")
        stats_res = friedman_posthoc(instances_used, score_table, args.alpha, args.control)
        stats_path = results_dir / "stats.md"
        write_stats_report(stats_res, stats_path, args.alpha)
        print(f"Stats report written to: {stats_path}")
        plot_cd_diagram(stats_res, results_dir / "stats_cd.png", args.alpha)

    # plotting
    instances_to_plot: List[str] = []
    if args.plot_all_instances:
        instances_to_plot = sorted({normalize_instance(r.get('instance_name') or r.get('instance') or r.get('problem_file')) for r in runs if (r.get('instance_name') or r.get('instance') or r.get('problem_file'))})
    elif args.plot_instances:
        instances_to_plot = args.plot_instances
    elif args.plot_instance:
        instances_to_plot = [args.plot_instance]

    for inst in instances_to_plot:
        out_path = plot_out if len(instances_to_plot) == 1 else results_dir / f"convergence_{normalize_instance(inst)}.png"
        plot_convergence(runs, inst, out_path, args.variants)


if __name__ == "__main__":
    main()
