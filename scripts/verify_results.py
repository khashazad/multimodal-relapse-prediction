"""Verify experiment results against known reference values.

Checks that re-run AUROC values match expected ± tolerance.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

CONFIGS_DIR = Path("configs")

# Reference values: best mean test AUROC from original runs.
# sweep = {} means no sweep filtering needed (single config or report overall best).
REFERENCES = {
    "ablation_baseline": {
        "auroc": 0.808,
        "sweep": {},
    },
    "ablation_focal": {
        "auroc": 0.845,
        "sweep": {"focal_gamma": 1.0, "focal_alpha": 0.5},
    },
    "ablation_focal_smooth": {
        "auroc": 0.838,
        "sweep": {"focal_gamma": 1.0, "focal_alpha": 0.3, "label_smoothing": 0.25},
    },
    "ablation_rope": {
        "auroc": 0.756,
        "sweep": {},
    },
    "ablation_recipe": {
        "auroc": 0.842,
        "sweep": {"optimizer": "adam", "lr_schedule": "none", "grad_clip": 0.0},
    },
}

TOLERANCE = 0.01


def load_config(name):
    with open(CONFIGS_DIR / f"{name}.json") as f:
        return json.load(f)


def load_results(name, output_dir):
    base = Path(output_dir)
    if not base.exists():
        return []
    results = []
    for fp in sorted(base.glob(f"{name}__*.json")):
        with open(fp) as f:
            results.append(json.load(f))
    return results


def get_sweep_value(result, param):
    for key in ("hyperparameters", "ablation_params"):
        if key in result and param in result[key]:
            return result[key][param]
    return result.get(param)


def filter_results(results, sweep_filter):
    """Keep only results matching all sweep_filter key=value pairs."""
    if not sweep_filter:
        return results
    filtered = []
    for r in results:
        match = all(
            get_sweep_value(r, k) == v for k, v in sweep_filter.items()
        )
        if match:
            filtered.append(r)
    return filtered


def mean_auroc(results):
    vals = [r.get("test_metrics", {}).get("auroc", float("nan")) for r in results]
    valid = [v for v in vals if not np.isnan(v)]
    return np.mean(valid) if valid else float("nan")


def main():
    print(f"{'Experiment':<30} {'Expected':>8} {'Actual':>8} {'Δ':>7} {'Status'}")
    print("-" * 70)

    all_pass = True
    for exp_name, ref in REFERENCES.items():
        config = load_config(exp_name)
        output_dir = config.get("executor", {}).get("output_dir", "outputs")
        results = load_results(exp_name, output_dir)

        if not results:
            print(f"{exp_name:<30} {ref['auroc']:>8.3f} {'—':>8} {'—':>7} MISSING")
            all_pass = False
            continue

        filtered = filter_results(results, ref["sweep"])
        if not filtered:
            sweep_str = ", ".join(f"{k}={v}" for k, v in ref["sweep"].items())
            print(f"{exp_name:<30} {ref['auroc']:>8.3f} {'—':>8} {'—':>7} NO MATCH ({sweep_str})")
            all_pass = False
            continue

        actual = mean_auroc(filtered)
        delta = actual - ref["auroc"]
        passed = abs(delta) <= TOLERANCE

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"{exp_name:<30} {ref['auroc']:>8.3f} {actual:>8.3f} {delta:>+7.3f} {status}")

    print("-" * 70)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
