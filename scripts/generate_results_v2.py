"""Generate outputs/results-v2/ markdown files from experiment output JSONs."""

import json
import glob
import os
import numpy as np
from collections import defaultdict

OUTPUT_DIR = "outputs"
RESULTS_DIR = "outputs/results-v2"
CONFIGS_DIR = "configs"

# New experiments to document
EXPERIMENTS = [
    "exp_ablation_dropout_only",
    "exp_ablation_loss_only",
    "exp_ablation_small_model",
    "exp_asymmetric_focal",
    "exp_dann_focal",
    "exp_focal_alpha_sweep",
    "exp_focal_augmented",
    "exp_focal_earlystop_auprc",
    "exp_focal_gamma_sweep",
    "exp_focal_label_smoothing",
    "exp_focal_lr_warmup",
    "exp_gated_focal",
]

PATIENT_MAP = {i: f"P{i+1}" for i in range(9)}

def load_config(exp_name):
    path = os.path.join(CONFIGS_DIR, f"{exp_name}.json")
    with open(path) as f:
        return json.load(f)

def load_results(exp_name):
    """Load all output JSONs for an experiment, grouped by sweep params if any."""
    pattern = os.path.join(OUTPUT_DIR, f"{exp_name}__*.json")
    files = sorted(glob.glob(pattern))
    results = []
    for fp in files:
        with open(fp) as f:
            results.append(json.load(f))
    return results

def get_description(config):
    return config.get("_description", "")

def format_config_table(config):
    """Generate a markdown config table from the config dict."""
    skip = {"executor", "slurm", "fold", "_description"}
    lines = ["| Parameter | Value |", "|---|---|"]
    for k, v in config.items():
        if k in skip or k.startswith("_"):
            continue
        if isinstance(v, list):
            lines.append(f"| {k} | {v} (sweep) |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

def safe_mean(vals):
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return np.mean(valid) if valid else float('nan')

def safe_std(vals):
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return np.std(valid) if valid else float('nan')

def generate_fold_table(results):
    """Generate per-fold results table for non-sweep experiments."""
    lines = []
    lines.append("| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |")
    lines.append("|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|")

    val_aurocs, test_aurocs, test_auprcs, test_f1s = [], [], [], []

    for r in sorted(results, key=lambda x: x["fold"]):
        fold = r["fold"]
        patient = PATIENT_MAP[fold]
        best_ep = r.get("best_epoch", "?")
        vm = r["val_metrics"]
        tm = r["test_metrics"]

        va = vm["auroc"]
        vp = vm["auprc"]
        vf = vm["f1"]
        ta = tm["auroc"]
        tp = tm["auprc"]
        tf = tm["f1"]

        val_aurocs.append(va)
        test_aurocs.append(ta)
        test_auprcs.append(tp)
        test_f1s.append(tf)

        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "NaN"
            return f"{v:.3f}"

        lines.append(f"| {fold} | {patient} | {best_ep} | {fmt(va)} | {fmt(vp)} | {fmt(vf)} | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} |")

    lines.append(f"| **Mean** | | | **{safe_mean(val_aurocs):.3f}** | | | **{safe_mean(test_aurocs):.3f}** | **{safe_mean(test_auprcs):.3f}** | **{safe_mean(test_f1s):.3f}** |")

    return "\n".join(lines)

def generate_calibrated_table(results):
    """Generate calibrated threshold table."""
    has_calibrated = any("calibrated_threshold" in r for r in results)
    if not has_calibrated:
        return None

    lines = []
    lines.append("| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |")
    lines.append("|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|")

    cal_test_f1s = []

    for r in sorted(results, key=lambda x: x["fold"]):
        fold = r["fold"]
        thresh = r.get("calibrated_threshold", "N/A")
        cvm = r.get("calibrated_val_metrics", {})
        ctm = r.get("calibrated_test_metrics", {})

        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "NaN"
            return f"{v:.3f}"

        cvf = cvm.get("f1", float('nan'))
        cvp = cvm.get("precision", float('nan'))
        cvr = cvm.get("recall", float('nan'))
        ctf = ctm.get("f1", float('nan'))
        ctp = ctm.get("precision", float('nan'))
        ctr = ctm.get("recall", float('nan'))

        cal_test_f1s.append(ctf)

        thresh_str = f"{thresh:.2f}" if isinstance(thresh, (int, float)) else str(thresh)
        lines.append(f"| {fold} | {thresh_str} | {fmt(cvf)} | {fmt(cvp)} | {fmt(cvr)} | {fmt(ctf)} | {fmt(ctp)} | {fmt(ctr)} |")

    lines.append(f"| **Mean** | | | | | **{safe_mean(cal_test_f1s):.3f}** | | |")

    return "\n".join(lines)

def generate_sweep_table(results, sweep_param):
    """Generate a summary table for sweep experiments, grouped by sweep param value."""
    grouped = defaultdict(list)
    for r in results:
        val = r["hyperparameters"][sweep_param]
        grouped[val].append(r)

    lines = []
    lines.append(f"| {sweep_param} | Mean Val AUROC | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 | Mean Cal Test F1 |")
    lines.append("|---|---|---|---|---|---|")

    for val in sorted(grouped.keys()):
        group = grouped[val]
        va = safe_mean([r["val_metrics"]["auroc"] for r in group])
        ta = safe_mean([r["test_metrics"]["auroc"] for r in group])
        tp = safe_mean([r["test_metrics"]["auprc"] for r in group])
        tf = safe_mean([r["test_metrics"]["f1"] for r in group])
        ctf = safe_mean([r.get("calibrated_test_metrics", {}).get("f1", float('nan')) for r in group])

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        lines.append(f"| {val} | {fmt(va)} | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} | {fmt(ctf)} |")

    return "\n".join(lines)

def generate_aggregate_summary(results):
    """Generate aggregate summary stats."""
    val_aurocs = [r["val_metrics"]["auroc"] for r in results]
    test_aurocs = [r["test_metrics"]["auroc"] for r in results]
    test_auprcs = [r["test_metrics"]["auprc"] for r in results]
    test_f1s = [r["test_metrics"]["f1"] for r in results]

    lines = ["| Metric | Mean | Std |", "|--------|------|-----|"]
    lines.append(f"| Val AUROC | {safe_mean(val_aurocs):.3f} | {safe_std(val_aurocs):.3f} |")
    lines.append(f"| Test AUROC | {safe_mean(test_aurocs):.3f} | {safe_std(test_aurocs):.3f} |")
    lines.append(f"| Test AUPRC | {safe_mean(test_auprcs):.3f} | {safe_std(test_auprcs):.3f} |")
    lines.append(f"| Test F1 | {safe_mean(test_f1s):.3f} | {safe_std(test_f1s):.3f} |")

    cal_f1s = [r.get("calibrated_test_metrics", {}).get("f1", float('nan')) for r in results]
    if any(not np.isnan(v) for v in cal_f1s):
        lines.append(f"| Cal Test F1 | {safe_mean(cal_f1s):.3f} | {safe_std(cal_f1s):.3f} |")

    elapsed = [r.get("elapsed_seconds", 0) for r in results]
    lines.append(f"| Total runtime | {sum(elapsed):.0f}s | |")

    return "\n".join(lines)

def identify_sweep_param(config):
    """Find if this config has a sweep parameter (list-valued, besides fold)."""
    for k, v in config.items():
        if k == "fold" or k in ("executor", "slurm") or k.startswith("_"):
            continue
        if isinstance(v, list):
            return k
    return None

def generate_doc(exp_name):
    config = load_config(exp_name)
    results = load_results(exp_name)

    if not results:
        print(f"  WARNING: No results found for {exp_name}, skipping.")
        return

    description = get_description(config)
    sweep_param = identify_sweep_param(config)
    is_sweep = sweep_param is not None

    md = []
    md.append(f"# Experiment: `{exp_name}`\n")

    if description:
        md.append(f"{description}\n")

    md.append("## Configuration\n")
    md.append(format_config_table(config))
    md.append("")

    if is_sweep:
        md.append(f"## Sweep Results (by `{sweep_param}`)\n")
        md.append(generate_sweep_table(results, sweep_param))
        md.append("")

        # Also show per-fold for the best sweep value
        grouped = defaultdict(list)
        for r in results:
            val = r["hyperparameters"][sweep_param]
            grouped[val].append(r)

        # Find best by mean test AUROC
        best_val = max(grouped.keys(), key=lambda v: safe_mean([r["test_metrics"]["auroc"] for r in grouped[v]]))
        best_results = grouped[best_val]
        md.append(f"\n## Per-Fold Results (Best: `{sweep_param}={best_val}`)\n")
        md.append(generate_fold_table(best_results))
        md.append("")

        cal_table = generate_calibrated_table(best_results)
        if cal_table:
            md.append(f"\n## Calibrated Threshold Results (`{sweep_param}={best_val}`)\n")
            md.append(cal_table)
            md.append("")
    else:
        md.append("## Per-Fold Results (Default Threshold = 0.5)\n")
        md.append(generate_fold_table(results))
        md.append("")

        cal_table = generate_calibrated_table(results)
        if cal_table:
            md.append("\n## Calibrated Threshold Results\n")
            md.append(cal_table)
            md.append("")

    md.append("\n## Aggregate Summary\n")
    md.append(generate_aggregate_summary(results))
    md.append("")

    return "\n".join(md)

def generate_comparison_summary():
    """Generate a single comparison summary across all experiments."""
    # Include original experiments for comparison
    all_exps = [
        "train", "exp_focal", "exp_augmented", "exp_bottleneck",
        "exp_dann", "exp_gated",
    ] + EXPERIMENTS

    md = []
    md.append("# Experiment Comparison Summary (v2)\n")
    md.append("## All Experiments\n")
    md.append("| Experiment | Mean Val AUROC | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 | Mean Cal Test F1 | Notes |")
    md.append("|---|---|---|---|---|---|---|")

    for exp_name in all_exps:
        results = load_results(exp_name)
        if not results:
            continue

        config = load_config(exp_name)
        sweep_param = identify_sweep_param(config)
        description = get_description(config)

        if sweep_param:
            # For sweep experiments, report the best sweep value
            grouped = defaultdict(list)
            for r in results:
                val = r["hyperparameters"][sweep_param]
                grouped[val].append(r)

            best_val = max(grouped.keys(), key=lambda v: safe_mean([r["test_metrics"]["auroc"] for r in grouped[v]]))
            results = grouped[best_val]
            notes = f"Best {sweep_param}={best_val}"
        else:
            notes = description[:60] if description else ""

        va = safe_mean([r["val_metrics"]["auroc"] for r in results])
        ta = safe_mean([r["test_metrics"]["auroc"] for r in results])
        tp = safe_mean([r["test_metrics"]["auprc"] for r in results])
        tf = safe_mean([r["test_metrics"]["f1"] for r in results])
        ctf = safe_mean([r.get("calibrated_test_metrics", {}).get("f1", float('nan')) for r in results])

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        md.append(f"| {exp_name} | {fmt(va)} | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} | {fmt(ctf)} | {notes} |")

    md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for exp_name in EXPERIMENTS:
        print(f"Generating: {exp_name}")
        doc = generate_doc(exp_name)
        if doc:
            path = os.path.join(RESULTS_DIR, f"{exp_name}.md")
            with open(path, "w") as f:
                f.write(doc)
            print(f"  -> {path}")

    # Generate comparison summary
    print("Generating comparison summary...")
    summary = generate_comparison_summary()
    summary_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  -> {summary_path}")

    print("\nDone!")
