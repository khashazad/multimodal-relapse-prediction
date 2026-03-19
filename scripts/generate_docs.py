"""Generate docs/results/ markdown from experiment output JSONs.

Auto-discovers all experiment configs, handles both exploration (src/train.py)
and ablation (src/ablation.py) ecosystems, supports multi-param sweeps.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

CONFIGS_DIR = Path("configs")
RESULTS_DIR = Path("docs/results")

PATIENT_MAP = {i: f"P{i + 1}" for i in range(9)}

# Phase categorization by config prefix
PHASE_ORDER = ["Baseline", "Exploration (V1)", "Ablation V1", "Ablation V2"]
PHASE_MAP = {
    "train": "Baseline",
    "exp_": "Exploration (V1)",
    "ablation_": "Ablation V1",
    "ablv2_": "Ablation V2",
}

SKIP_CONFIGS = {"preprocessing"}


def get_phase(name):
    for prefix, phase in PHASE_MAP.items():
        if name == prefix.rstrip("_") or name.startswith(prefix):
            return phase
    return "Other"


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.3f}"


def safe_mean(vals):
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return np.mean(valid) if valid else float("nan")


def safe_std(vals):
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return np.std(valid) if valid else float("nan")


# ---------------------------------------------------------------------------
# Config & result loading
# ---------------------------------------------------------------------------


def discover_experiments():
    """Find all experiment configs (excluding preprocessing and non-JSON)."""
    experiments = []
    for p in sorted(CONFIGS_DIR.glob("*.json")):
        name = p.stem
        if name in SKIP_CONFIGS:
            continue
        # Must have an executor section
        data = json.loads(p.read_text())
        if "executor" not in data:
            continue
        experiments.append(name)
    return experiments


def load_config(exp_name):
    path = CONFIGS_DIR / f"{exp_name}.json"
    with open(path) as f:
        return json.load(f)


def load_results(exp_name, output_dir):
    """Load all output JSONs for an experiment from its output_dir."""
    base = Path(output_dir)
    if not base.exists():
        return []
    files = sorted(base.glob(f"{exp_name}__*.json"))
    results = []
    for fp in files:
        with open(fp) as f:
            results.append(json.load(f))
    return results


def get_sweep_value(result, param):
    """Get a sweep parameter value from either hyperparameters or ablation_params."""
    for key in ("hyperparameters", "ablation_params"):
        if key in result and param in result[key]:
            return result[key][param]
    # Also check top-level (some outputs store fold at top level)
    return result.get(param)


def identify_sweep_params(config):
    """Find all sweep parameters (list-valued, besides fold)."""
    sweeps = []
    for k, v in config.items():
        if k == "fold" or k in ("executor", "slurm") or k.startswith("_"):
            continue
        if isinstance(v, list):
            sweeps.append(k)
    return sweeps


def get_description(config):
    return config.get("_description", "")


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_by_sweep(results, sweep_params):
    """Group results by a tuple of sweep parameter values."""
    grouped = defaultdict(list)
    for r in results:
        key = tuple(get_sweep_value(r, p) for p in sweep_params)
        grouped[key].append(r)
    return grouped


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------


def format_config_table(config):
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


def get_val_metric(r, metric):
    vm = r.get("val_metrics")
    if vm is None:
        return float("nan")
    return vm.get(metric, float("nan"))


def get_test_metric(r, metric):
    tm = r.get("test_metrics", {})
    return tm.get(metric, float("nan"))


def generate_fold_table(results, has_val=True):
    """Generate per-fold results table."""
    lines = []
    if has_val:
        lines.append(
            "| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |"
        )
        lines.append(
            "|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|"
        )
    else:
        lines.append(
            "| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |"
        )
        lines.append(
            "|------|---------|------------|------------|---------|"
        )

    test_aurocs, test_auprcs, test_f1s = [], [], []
    val_aurocs = []

    for r in sorted(results, key=lambda x: x.get("fold", 0)):
        fold = r.get("fold", "?")
        patient = PATIENT_MAP.get(fold, f"P{fold}")
        ta = get_test_metric(r, "auroc")
        tp = get_test_metric(r, "auprc")
        tf = get_test_metric(r, "f1")
        test_aurocs.append(ta)
        test_auprcs.append(tp)
        test_f1s.append(tf)

        if has_val:
            best_ep = r.get("best_epoch", "?")
            va = get_val_metric(r, "auroc")
            vp = get_val_metric(r, "auprc")
            vf = get_val_metric(r, "f1")
            val_aurocs.append(va)
            lines.append(
                f"| {fold} | {patient} | {best_ep} | {fmt(va)} | {fmt(vp)} | {fmt(vf)} | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} |"
            )
        else:
            lines.append(
                f"| {fold} | {patient} | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} |"
            )

    if has_val:
        lines.append(
            f"| **Mean** | | | **{fmt(safe_mean(val_aurocs))}** | | | **{fmt(safe_mean(test_aurocs))}** | **{fmt(safe_mean(test_auprcs))}** | **{fmt(safe_mean(test_f1s))}** |"
        )
    else:
        lines.append(
            f"| **Mean** | | **{fmt(safe_mean(test_aurocs))}** | **{fmt(safe_mean(test_auprcs))}** | **{fmt(safe_mean(test_f1s))}** |"
        )

    return "\n".join(lines)


def generate_calibrated_table(results):
    has_calibrated = any("calibrated_threshold" in r for r in results)
    if not has_calibrated:
        return None

    lines = [
        "| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |",
        "|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|",
    ]

    cal_test_f1s = []
    for r in sorted(results, key=lambda x: x.get("fold", 0)):
        fold = r.get("fold", "?")
        thresh = r.get("calibrated_threshold", "N/A")
        cvm = r.get("calibrated_val_metrics", {})
        ctm = r.get("calibrated_test_metrics", {})

        ctf = ctm.get("f1", float("nan"))
        cal_test_f1s.append(ctf)

        thresh_str = f"{thresh:.2f}" if isinstance(thresh, (int, float)) else str(thresh)
        lines.append(
            f"| {fold} | {thresh_str} | {fmt(cvm.get('f1', float('nan')))} | {fmt(cvm.get('precision', float('nan')))} | {fmt(cvm.get('recall', float('nan')))} | {fmt(ctf)} | {fmt(ctm.get('precision', float('nan')))} | {fmt(ctm.get('recall', float('nan')))} |"
        )

    lines.append(f"| **Mean** | | | | | **{fmt(safe_mean(cal_test_f1s))}** | | |")
    return "\n".join(lines)


def generate_sweep_table(results, sweep_params, has_val=True):
    """Summary table grouped by sweep param combo."""
    grouped = group_by_sweep(results, sweep_params)

    header_params = " | ".join(f"`{p}`" for p in sweep_params)
    if has_val:
        lines = [
            f"| {header_params} | Mean Val AUROC | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |",
            "|" + "---|" * (len(sweep_params) + 4),
        ]
    else:
        lines = [
            f"| {header_params} | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |",
            "|" + "---|" * (len(sweep_params) + 3),
        ]

    # Sort by mean test AUROC descending
    sorted_keys = sorted(
        grouped.keys(),
        key=lambda k: safe_mean([get_test_metric(r, "auroc") for r in grouped[k]]),
        reverse=True,
    )

    for key in sorted_keys:
        group = grouped[key]
        ta = safe_mean([get_test_metric(r, "auroc") for r in group])
        tp = safe_mean([get_test_metric(r, "auprc") for r in group])
        tf = safe_mean([get_test_metric(r, "f1") for r in group])

        vals = " | ".join(str(v) for v in key)
        if has_val:
            va = safe_mean([get_val_metric(r, "auroc") for r in group])
            lines.append(f"| {vals} | {fmt(va)} | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} |")
        else:
            lines.append(f"| {vals} | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} |")

    return "\n".join(lines)


def generate_aggregate_summary(results, has_val=True):
    test_aurocs = [get_test_metric(r, "auroc") for r in results]
    test_auprcs = [get_test_metric(r, "auprc") for r in results]
    test_f1s = [get_test_metric(r, "f1") for r in results]

    lines = ["| Metric | Mean | Std |", "|--------|------|-----|"]

    if has_val:
        val_aurocs = [get_val_metric(r, "auroc") for r in results]
        lines.append(f"| Val AUROC | {fmt(safe_mean(val_aurocs))} | {fmt(safe_std(val_aurocs))} |")

    lines.append(f"| Test AUROC | {fmt(safe_mean(test_aurocs))} | {fmt(safe_std(test_aurocs))} |")
    lines.append(f"| Test AUPRC | {fmt(safe_mean(test_auprcs))} | {fmt(safe_std(test_auprcs))} |")
    lines.append(f"| Test F1 | {fmt(safe_mean(test_f1s))} | {fmt(safe_std(test_f1s))} |")

    cal_f1s = [r.get("calibrated_test_metrics", {}).get("f1", float("nan")) for r in results]
    if any(not np.isnan(v) for v in cal_f1s if isinstance(v, float)):
        lines.append(f"| Cal Test F1 | {fmt(safe_mean(cal_f1s))} | {fmt(safe_std(cal_f1s))} |")

    elapsed = [r.get("elapsed_seconds", 0) for r in results]
    if any(e > 0 for e in elapsed):
        lines.append(f"| Total runtime | {sum(elapsed):.0f}s | |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------


def generate_doc(exp_name):
    config = load_config(exp_name)
    output_dir = config.get("executor", {}).get("output_dir", "outputs")
    results = load_results(exp_name, output_dir)

    if not results:
        print(f"  WARNING: no results for {exp_name}, skipping")
        return None

    description = get_description(config)
    sweep_params = identify_sweep_params(config)
    is_sweep = len(sweep_params) > 0
    has_val = any("val_metrics" in r for r in results)

    md = [f"# Experiment: `{exp_name}`\n"]
    if description:
        md.append(f"{description}\n")

    md.append("## Configuration\n")
    md.append(format_config_table(config))
    md.append("")

    if is_sweep:
        params_str = ", ".join(f"`{p}`" for p in sweep_params)
        md.append(f"\n## Sweep Results (by {params_str})\n")
        md.append(generate_sweep_table(results, sweep_params, has_val))
        md.append("")

        # Per-fold for best combo
        grouped = group_by_sweep(results, sweep_params)
        best_key = max(
            grouped.keys(),
            key=lambda k: safe_mean([get_test_metric(r, "auroc") for r in grouped[k]]),
        )
        best_results = grouped[best_key]
        combo_str = ", ".join(f"{p}={v}" for p, v in zip(sweep_params, best_key))
        md.append(f"\n## Per-Fold Results (Best: {combo_str})\n")
        md.append(generate_fold_table(best_results, has_val))
        md.append("")

        cal_table = generate_calibrated_table(best_results)
        if cal_table:
            md.append(f"\n## Calibrated Threshold Results ({combo_str})\n")
            md.append(cal_table)
            md.append("")
    else:
        md.append("\n## Per-Fold Results\n")
        md.append(generate_fold_table(results, has_val))
        md.append("")

        cal_table = generate_calibrated_table(results)
        if cal_table:
            md.append("\n## Calibrated Threshold Results\n")
            md.append(cal_table)
            md.append("")

    md.append("\n## Aggregate Summary\n")
    agg_results = best_results if is_sweep else results
    md.append(generate_aggregate_summary(agg_results, has_val))
    md.append("")

    return "\n".join(md)


def generate_summary(experiments):
    """Leaderboard grouped by phase."""
    md = ["# Results Summary\n"]
    md.append("Auto-generated from experiment output JSONs.\n")

    for phase in PHASE_ORDER:
        phase_exps = [e for e in experiments if get_phase(e) == phase]
        if not phase_exps:
            continue

        md.append(f"\n## {phase}\n")
        md.append(
            "| Experiment | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 | Best Sweep | Notes |"
        )
        md.append("|---|---|---|---|---|---|")

        for exp_name in phase_exps:
            config = load_config(exp_name)
            output_dir = config.get("executor", {}).get("output_dir", "outputs")
            results = load_results(exp_name, output_dir)

            if not results:
                md.append(f"| {exp_name} | — | — | — | — | no results |")
                continue

            sweep_params = identify_sweep_params(config)
            description = get_description(config)
            notes = description[:60] if description else ""
            sweep_str = ""

            if sweep_params:
                grouped = group_by_sweep(results, sweep_params)
                best_key = max(
                    grouped.keys(),
                    key=lambda k: safe_mean([get_test_metric(r, "auroc") for r in grouped[k]]),
                )
                results = grouped[best_key]
                sweep_str = ", ".join(f"{p}={v}" for p, v in zip(sweep_params, best_key))

            ta = safe_mean([get_test_metric(r, "auroc") for r in results])
            tp = safe_mean([get_test_metric(r, "auprc") for r in results])
            tf = safe_mean([get_test_metric(r, "f1") for r in results])

            md.append(
                f"| [{exp_name}]({exp_name}.md) | {fmt(ta)} | {fmt(tp)} | {fmt(tf)} | {sweep_str} | {notes} |"
            )

    md.append("")
    return "\n".join(md)


def generate_index(experiments):
    """README.md index with links to all per-experiment docs."""
    md = ["# Experiment Results\n"]
    md.append("Auto-generated documentation for all experiments.\n")
    md.append("- [Summary / Leaderboard](summary.md)\n")

    for phase in PHASE_ORDER:
        phase_exps = [e for e in experiments if get_phase(e) == phase]
        if not phase_exps:
            continue
        md.append(f"\n## {phase}\n")
        for exp_name in phase_exps:
            md.append(f"- [{exp_name}]({exp_name}.md)")

    md.append("")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    experiments = discover_experiments()
    print(f"Discovered {len(experiments)} experiments: {', '.join(experiments)}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    for exp_name in experiments:
        print(f"Generating: {exp_name}")
        doc = generate_doc(exp_name)
        if doc:
            path = RESULTS_DIR / f"{exp_name}.md"
            path.write_text(doc)
            print(f"  -> {path}")
            generated += 1

    # Summary leaderboard
    print("\nGenerating summary...")
    summary = generate_summary(experiments)
    (RESULTS_DIR / "summary.md").write_text(summary)
    print(f"  -> {RESULTS_DIR / 'summary.md'}")

    # Index
    index = generate_index(experiments)
    (RESULTS_DIR / "README.md").write_text(index)
    print(f"  -> {RESULTS_DIR / 'README.md'}")

    print(f"\nDone! {generated}/{len(experiments)} experiments documented.")
