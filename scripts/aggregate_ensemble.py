"""
Post-hoc ensemble aggregation and analysis.

Loads base model predictions from outputs/ensemble_phase/ and computes:
  1. Simple average
  2. Weighted average (optimized on val AUROC)
  3. Rank average
  4. Stacking (logistic regression, OOF via LOPO)

Usage:
  python scripts/aggregate_ensemble.py --experiment stacking --output_dir outputs/ensemble_phase
  python scripts/aggregate_ensemble.py --experiment all --output_dir outputs/ensemble_phase
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.stats import wilcoxon, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


PATIENTS = [f"P{i}" for i in range(1, 10)]

DIAGNOSIS_GROUPS = {
    "bipolar": ["P3", "P4", "P5", "P6", "P8", "P9"],
    "schiz-spectrum": ["P2", "P7"],
    "brief-psychotic": ["P1"],
}


def load_results(output_dir: str, pattern: str = "") -> list[dict]:
    """Load all JSON result files matching pattern from output_dir."""
    results = []
    out_path = Path(output_dir)
    if not out_path.exists():
        print(f"Warning: {output_dir} does not exist")
        return results
    for f in sorted(out_path.glob("*.json")):
        if pattern and pattern not in f.name:
            continue
        with open(f) as fh:
            try:
                results.append(json.load(fh))
            except json.JSONDecodeError:
                print(f"Warning: could not parse {f}")
    return results


def group_by_fold(results: list[dict]) -> dict[int, list[dict]]:
    """Group results by fold number."""
    by_fold = {}
    for r in results:
        fold = r.get("fold")
        if fold is not None:
            by_fold.setdefault(fold, []).append(r)
    return by_fold


def get_predictions(result: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract (y_true, y_score) from a result dict."""
    y_true = result.get("y_true", [])
    y_score = result.get("y_score", [])
    if not y_true or not y_score or len(y_true) != len(y_score):
        return None
    return np.array(y_true), np.array(y_score)


def model_key(result: dict) -> str:
    """Generate a unique key for a model config (excluding fold)."""
    hp = result.get("hyperparameters", {})
    mt = hp.get("model_type", result.get("method", "unknown"))
    fs = result.get("feature_set", "unknown")
    dm = hp.get("d_model", hp.get("n_channels", ""))
    return f"{mt}_{fs}_{dm}"


# =============================================================================
# Ensemble methods
# =============================================================================

def simple_average(predictions_list: list[np.ndarray]) -> np.ndarray:
    return np.mean(predictions_list, axis=0)


def rank_average(predictions_list: list[np.ndarray]) -> np.ndarray:
    ranked = [rankdata(p) / len(p) for p in predictions_list]
    return np.mean(ranked, axis=0)


def weighted_average_optimize(
    predictions_list: list[np.ndarray],
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find optimal weights to maximize AUROC."""
    n_models = len(predictions_list)
    preds_matrix = np.stack(predictions_list)  # (n_models, n_samples)

    def neg_auroc(weights):
        w = np.abs(weights)
        w = w / w.sum()
        ensemble = w @ preds_matrix
        try:
            return -roc_auc_score(y_true, ensemble)
        except Exception:
            return 0.0

    init_w = np.ones(n_models) / n_models
    result = optimize.minimize(neg_auroc, init_w, method="Nelder-Mead",
                               options={"maxiter": 1000})
    best_w = np.abs(result.x)
    best_w = best_w / best_w.sum()
    ensemble = best_w @ preds_matrix
    return ensemble, best_w


def stacking_lopo(
    fold_model_preds: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Stacking via logistic regression with LOPO.

    For fold k, train meta-learner on folds != k predictions, predict on fold k.
    Returns {fold: (y_true, y_score_stacked, y_score_avg)}.
    """
    all_folds = sorted(fold_model_preds.keys())
    model_keys = None

    # Verify all folds have same models
    for fold in all_folds:
        keys = sorted(fold_model_preds[fold].keys())
        if model_keys is None:
            model_keys = keys
        elif keys != model_keys:
            # Use intersection
            model_keys = sorted(set(model_keys) & set(keys))

    if not model_keys:
        print("Warning: no common models across folds for stacking")
        return {}

    print(f"  Stacking with {len(model_keys)} base models across {len(all_folds)} folds")

    results = {}
    for test_fold in all_folds:
        # Build meta-features for train folds
        X_meta_train, y_meta_train = [], []
        for train_fold in all_folds:
            if train_fold == test_fold:
                continue
            preds_row = []
            y_true_fold = None
            for mk in model_keys:
                if mk not in fold_model_preds[train_fold]:
                    continue
                yt, ys = fold_model_preds[train_fold][mk]
                preds_row.append(ys)
                if y_true_fold is None:
                    y_true_fold = yt
            if preds_row and y_true_fold is not None:
                X_meta_train.append(np.stack(preds_row, axis=1))  # (n_samples, n_models)
                y_meta_train.append(y_true_fold)

        if not X_meta_train:
            continue

        X_meta_train = np.concatenate(X_meta_train, axis=0)
        y_meta_train = np.concatenate(y_meta_train, axis=0)

        # Build meta-features for test fold
        preds_test = []
        y_true_test = None
        for mk in model_keys:
            if mk not in fold_model_preds[test_fold]:
                continue
            yt, ys = fold_model_preds[test_fold][mk]
            preds_test.append(ys)
            if y_true_test is None:
                y_true_test = yt

        if not preds_test or y_true_test is None:
            continue

        X_meta_test = np.stack(preds_test, axis=1)

        # Train meta-learner
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta.fit(X_meta_train, y_meta_train)
        y_score_stacked = meta.predict_proba(X_meta_test)[:, 1]

        # Also compute simple average for comparison
        y_score_avg = np.mean(np.stack(preds_test), axis=0)

        results[test_fold] = (y_true_test, y_score_stacked, y_score_avg)

    return results


# =============================================================================
# Analysis helpers
# =============================================================================

def compute_metrics_table(
    fold_results: dict[int, tuple[np.ndarray, np.ndarray]],
    label: str,
) -> dict:
    """Compute per-fold and mean AUROC/AUPRC."""
    aurocs, auprcs = [], []
    per_fold = {}
    for fold in sorted(fold_results.keys()):
        y_true, y_score = fold_results[fold][:2]
        if len(np.unique(y_true)) < 2:
            continue
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        aurocs.append(auroc)
        auprcs.append(auprc)
        per_fold[fold] = {"auroc": auroc, "auprc": auprc,
                          "patient": PATIENTS[fold]}

    return {
        "label": label,
        "per_fold": per_fold,
        "mean_auroc": np.mean(aurocs) if aurocs else float("nan"),
        "std_auroc": np.std(aurocs) if aurocs else float("nan"),
        "mean_auprc": np.mean(auprcs) if auprcs else float("nan"),
        "std_auprc": np.std(auprcs) if auprcs else float("nan"),
        "n_folds": len(aurocs),
    }


def format_results_table(metrics_list: list[dict]) -> str:
    """Format metrics as markdown table."""
    lines = []
    lines.append("| Method | Mean AUROC | Std | Mean AUPRC | Std | N Folds |")
    lines.append("|--------|-----------|-----|-----------|-----|---------|")
    for m in metrics_list:
        lines.append(
            f"| {m['label']:<30s} | {m['mean_auroc']:.4f} | {m['std_auroc']:.4f} | "
            f"{m['mean_auprc']:.4f} | {m['std_auprc']:.4f} | {m['n_folds']} |"
        )
    return "\n".join(lines)


def per_fold_table(metrics_list: list[dict]) -> str:
    """Format per-fold AUROC comparison."""
    lines = []
    header = "| Fold | Patient |"
    sep = "|------|---------|"
    for m in metrics_list:
        header += f" {m['label'][:15]:>15s} |"
        sep += "----------------|"
    lines.append(header)
    lines.append(sep)

    for fold in range(9):
        row = f"| {fold}    | {PATIENTS[fold]}      |"
        for m in metrics_list:
            pf = m.get("per_fold", {}).get(fold, {})
            auroc = pf.get("auroc", float("nan"))
            row += f" {auroc:>15.4f} |"
        lines.append(row)

    return "\n".join(lines)


def wilcoxon_test(aurocs_a: list[float], aurocs_b: list[float], label_a: str, label_b: str) -> str:
    """Paired Wilcoxon signed-rank test."""
    if len(aurocs_a) != len(aurocs_b) or len(aurocs_a) < 5:
        return f"  Wilcoxon test: insufficient paired samples ({len(aurocs_a)})"
    try:
        stat, p_value = wilcoxon(aurocs_a, aurocs_b, alternative="greater")
        sig = "significant" if p_value < 0.05 else "not significant"
        return (f"  Wilcoxon ({label_a} > {label_b}): "
                f"stat={stat:.1f}, p={p_value:.4f} ({sig})")
    except Exception as e:
        return f"  Wilcoxon test failed: {e}"


def diagnosis_group_breakdown(
    fold_results: dict[int, tuple[np.ndarray, np.ndarray]],
    label: str,
) -> str:
    """Per-diagnosis-group AUROC."""
    lines = [f"\n### {label} — by diagnosis group"]
    for group_name, group_patients in DIAGNOSIS_GROUPS.items():
        group_aurocs = []
        for fold in sorted(fold_results.keys()):
            if PATIENTS[fold] in group_patients:
                y_true, y_score = fold_results[fold][:2]
                if len(np.unique(y_true)) >= 2:
                    group_aurocs.append(roc_auc_score(y_true, y_score))
        if group_aurocs:
            lines.append(f"  {group_name}: AUROC={np.mean(group_aurocs):.4f} "
                         f"(±{np.std(group_aurocs):.4f}, n={len(group_aurocs)})")
        else:
            lines.append(f"  {group_name}: no valid folds")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def run_analysis(output_dir: str, experiment: str, model_filter: str = "all"):
    print(f"Loading results from {output_dir}...")

    # Load all base model results
    all_results = load_results(output_dir)
    if not all_results:
        print("No results found!")
        return

    # Separate by method
    base_results = [r for r in all_results if r.get("method") == "base_model"]
    film_results = [r for r in all_results if r.get("method") == "film"]
    tta_results = [r for r in all_results if r.get("method") == "tta"]

    # Keep originals for subset analysis
    original_film_results = list(film_results)

    # Apply model filter
    if model_filter == "transformer":
        base_results = [r for r in base_results
                        if r.get("hyperparameters", {}).get("model_type") == "transformer"]
    elif model_filter == "transformer+film":
        base_results = [r for r in base_results
                        if r.get("hyperparameters", {}).get("model_type") == "transformer"]
        for r in film_results:
            r_copy = dict(r)
            r_copy["method"] = "base_model"
            base_results.append(r_copy)
        film_results = []

    print(f"  Base models: {len(base_results)} (filter={model_filter})")
    print(f"  FiLM: {len(film_results)}")
    print(f"  TTA: {len(tta_results)}")

    # Organize base model predictions: {fold: {model_key: (y_true, y_score)}}
    fold_model_preds = {}
    for r in base_results:
        preds = get_predictions(r)
        if preds is None:
            continue
        fold = r["fold"]
        mk = model_key(r)
        fold_model_preds.setdefault(fold, {})[mk] = preds

    all_metrics = []
    all_fold_results = {}

    # --- Single best baseline ---
    baseline_aurocs = {}
    for fold, models in fold_model_preds.items():
        best_auroc = -1
        best_preds = None
        for mk, (yt, ys) in models.items():
            if len(np.unique(yt)) < 2:
                continue
            a = roc_auc_score(yt, ys)
            if a > best_auroc:
                best_auroc = a
                best_preds = (yt, ys)
        if best_preds:
            baseline_aurocs[fold] = best_preds
    if baseline_aurocs:
        m = compute_metrics_table(baseline_aurocs, "Single best (per-fold)")
        all_metrics.append(m)
        all_fold_results["single_best"] = baseline_aurocs

    # --- Simple average ---
    if fold_model_preds and experiment in ("stacking", "all"):
        avg_results = {}
        for fold, models in fold_model_preds.items():
            preds_list = [ys for yt, ys in models.values()]
            y_true = list(models.values())[0][0]
            if len(preds_list) > 1:
                avg_pred = simple_average(preds_list)
                avg_results[fold] = (y_true, avg_pred)
        if avg_results:
            m = compute_metrics_table(avg_results, "Simple average")
            all_metrics.append(m)
            all_fold_results["simple_avg"] = avg_results

    # --- Rank average ---
    if fold_model_preds and experiment in ("stacking", "all"):
        rank_results = {}
        for fold, models in fold_model_preds.items():
            preds_list = [ys for yt, ys in models.values()]
            y_true = list(models.values())[0][0]
            if len(preds_list) > 1:
                rank_pred = rank_average(preds_list)
                rank_results[fold] = (y_true, rank_pred)
        if rank_results:
            m = compute_metrics_table(rank_results, "Rank average")
            all_metrics.append(m)
            all_fold_results["rank_avg"] = rank_results

    # --- Weighted average ---
    if fold_model_preds and experiment in ("stacking", "all"):
        wt_results = {}
        for fold, models in fold_model_preds.items():
            preds_list = [ys for yt, ys in models.values()]
            y_true = list(models.values())[0][0]
            if len(preds_list) > 1 and len(np.unique(y_true)) >= 2:
                wt_pred, weights = weighted_average_optimize(preds_list, y_true)
                wt_results[fold] = (y_true, wt_pred)
        if wt_results:
            m = compute_metrics_table(wt_results, "Weighted average")
            all_metrics.append(m)
            all_fold_results["weighted_avg"] = wt_results

    # --- Stacking ---
    if fold_model_preds and experiment in ("stacking", "all"):
        print("\nRunning stacking (logistic regression)...")
        stacking_results = stacking_lopo(fold_model_preds)
        if stacking_results:
            stack_fold_results = {
                fold: (yt, ys_stack)
                for fold, (yt, ys_stack, ys_avg) in stacking_results.items()
            }
            m = compute_metrics_table(stack_fold_results, "Stacking (LR)")
            all_metrics.append(m)
            all_fold_results["stacking"] = stack_fold_results

    # --- FiLM results ---
    if film_results and experiment in ("film", "all"):
        film_fold_results = {}
        for r in film_results:
            preds = get_predictions(r)
            if preds is None:
                continue
            fold = r["fold"]
            # Keep best FiLM per fold if multiple
            if fold in film_fold_results:
                existing_auroc = roc_auc_score(*film_fold_results[fold])
                new_auroc = roc_auc_score(*preds)
                if new_auroc <= existing_auroc:
                    continue
            film_fold_results[fold] = preds
        if film_fold_results:
            m = compute_metrics_table(film_fold_results, "FiLM transformer")
            all_metrics.append(m)
            all_fold_results["film"] = film_fold_results

    # --- TTA results ---
    if tta_results and experiment in ("tta", "all"):
        # Group by variant
        tta_variants = {}
        for r in tta_results:
            preds = get_predictions(r)
            if preds is None:
                continue
            variant = r.get("hyperparameters", {}).get("tta_variant", "unknown")
            tta_lr = r.get("hyperparameters", {}).get("tta_lr", 0)
            key = f"TTA_{variant}_lr{tta_lr}"
            tta_variants.setdefault(key, {})
            fold = r["fold"]
            # Keep best per fold
            if fold in tta_variants[key]:
                existing = roc_auc_score(*tta_variants[key][fold])
                new = roc_auc_score(*preds)
                if new <= existing:
                    continue
            tta_variants[key][fold] = preds

        for variant_key, fold_results in tta_variants.items():
            m = compute_metrics_table(fold_results, variant_key)
            all_metrics.append(m)
            all_fold_results[variant_key] = fold_results

    # --- Transformer subset analysis ---
    subset_metrics = []
    if model_filter in ("transformer", "transformer+film"):
        # Map model_key -> hyperparameters for filtering
        model_hp = {}
        for r in base_results:
            mk = model_key(r)
            model_hp[mk] = r.get("hyperparameters", {})

        subsets = [
            ("All 6 transformers", lambda hp: True),
            ("Top-4 (d>=512)", lambda hp: hp.get("d_model", 0) >= 512),
            ("Top-2 (d=1024)", lambda hp: hp.get("d_model", 0) == 1024),
        ]

        for subset_name, filter_fn in subsets:
            rank_results = {}
            avg_results = {}
            for fold, models in fold_model_preds.items():
                filtered = [(mk, yt, ys) for mk, (yt, ys) in models.items()
                            if mk in model_hp and filter_fn(model_hp[mk])]
                if len(filtered) > 1:
                    preds_list = [ys for _, _, ys in filtered]
                    y_true = filtered[0][1]
                    rank_results[fold] = (y_true, rank_average(preds_list))
                    avg_results[fold] = (y_true, simple_average(preds_list))
            n_models = len([mk for mk in model_hp if filter_fn(model_hp[mk])])
            if rank_results:
                m_rank = compute_metrics_table(rank_results,
                                               f"Rank: {subset_name} ({n_models})")
                m_avg = compute_metrics_table(avg_results,
                                              f"Avg: {subset_name} ({n_models})")
                subset_metrics.extend([m_rank, m_avg])
                all_fold_results[f"subset_rank_{subset_name}"] = rank_results
                all_fold_results[f"subset_avg_{subset_name}"] = avg_results

        # Transformers + FiLM subset
        if original_film_results:
            film_fold_preds = {}
            for r in original_film_results:
                preds = get_predictions(r)
                if preds is None:
                    continue
                fold = r["fold"]
                mk = f"film_{model_key(r)}"
                film_fold_preds.setdefault(fold, {})[mk] = preds

            rank_results_tf = {}
            avg_results_tf = {}
            for fold in fold_model_preds:
                all_preds = [ys for yt, ys in fold_model_preds[fold].values()]
                if fold in film_fold_preds:
                    all_preds.extend([ys for yt, ys in film_fold_preds[fold].values()])
                y_true = list(fold_model_preds[fold].values())[0][0]
                if len(all_preds) > 1:
                    rank_results_tf[fold] = (y_true, rank_average(all_preds))
                    avg_results_tf[fold] = (y_true, simple_average(all_preds))

            n_film = len(set(f"film_{model_key(r)}" for r in original_film_results))
            total = len(model_hp) + n_film
            if rank_results_tf:
                m_rank = compute_metrics_table(rank_results_tf,
                                               f"Rank: Trans+FiLM ({total})")
                m_avg = compute_metrics_table(avg_results_tf,
                                              f"Avg: Trans+FiLM ({total})")
                subset_metrics.extend([m_rank, m_avg])
                all_fold_results["subset_rank_trans_film"] = rank_results_tf
                all_fold_results["subset_avg_trans_film"] = avg_results_tf

    # --- Generate report ---
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)

    if model_filter == "transformer":
        title = "# 015b — Transformer-Only Ensemble Results\n"
    elif model_filter == "transformer+film":
        title = "# 015b — Transformer+FiLM Ensemble Results\n"
    else:
        title = "# 015 — Ensemble Experiment Results\n"
    report_lines = [title]

    if all_metrics:
        report_lines.append("## Summary\n")
        report_lines.append(format_results_table(all_metrics))
        report_lines.append("")

        report_lines.append("\n## Per-Fold AUROC\n")
        report_lines.append(per_fold_table(all_metrics))
        report_lines.append("")

    # Wilcoxon tests
    if len(all_metrics) >= 2:
        report_lines.append("\n## Statistical Tests\n")
        baseline_name = all_metrics[0]["label"]
        baseline_per = all_metrics[0].get("per_fold", {})
        for m in all_metrics[1:]:
            comp_per = m.get("per_fold", {})
            common_folds = sorted(set(baseline_per.keys()) & set(comp_per.keys()))
            if len(common_folds) >= 5:
                a = [baseline_per[f]["auroc"] for f in common_folds]
                b = [comp_per[f]["auroc"] for f in common_folds]
                report_lines.append(
                    wilcoxon_test(b, a, m["label"], baseline_name)
                )

    # Transformer subset analysis section
    if subset_metrics:
        report_lines.append("\n## Transformer Subset Analysis\n")
        report_lines.append(format_results_table(subset_metrics))
        report_lines.append("")
        report_lines.append("\n### Subset Per-Fold AUROC\n")
        # Only show rank-average rows in per-fold table
        rank_subset_metrics = [m for m in subset_metrics if m["label"].startswith("Rank:")]
        report_lines.append(per_fold_table(rank_subset_metrics))
        report_lines.append("")

        # Wilcoxon: best subset vs single best baseline
        if all_metrics and rank_subset_metrics:
            report_lines.append("\n### Subset vs Baseline (Wilcoxon)\n")
            baseline_per = all_metrics[0].get("per_fold", {})
            for m in rank_subset_metrics:
                comp_per = m.get("per_fold", {})
                common_folds = sorted(set(baseline_per.keys()) & set(comp_per.keys()))
                if len(common_folds) >= 5:
                    a = [baseline_per[f]["auroc"] for f in common_folds]
                    b = [comp_per[f]["auroc"] for f in common_folds]
                    report_lines.append(
                        wilcoxon_test(b, a, m["label"], all_metrics[0]["label"])
                    )

    # Diagnosis group breakdown
    report_lines.append("\n## Diagnosis Group Breakdown\n")
    for key, fold_results in all_fold_results.items():
        report_lines.append(diagnosis_group_breakdown(fold_results, key))

    report = "\n".join(report_lines)
    print(report)

    # Save report
    report_path = "015_ensemble_results.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble aggregation and analysis")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["stacking", "film", "tta", "all"],
                        help="Which experiment to analyze")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/ensemble_phase",
                        help="Directory with result JSONs")
    parser.add_argument("--model_filter", type=str, default="all",
                        choices=["all", "transformer", "transformer+film"],
                        help="Filter base models before ensembling")
    args = parser.parse_args()
    run_analysis(args.output_dir, args.experiment, args.model_filter)
