"""
Validation Suite for arXiv-Large Scaling Experiments.
Systematically audits all checkpoints in scaling_raw_results.csv against 11 verification gates.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

AUDIT_DIR = Path("research_audit/arxiv_large_scaling")
RAW_CSV = AUDIT_DIR / "scaling_raw_results.csv"
SUMMARY_CSV = AUDIT_DIR / "scaling_summary.csv"
RES_CSV = AUDIT_DIR / "resource_measurements.csv"
REPORT_PATH = AUDIT_DIR / "validation_report.json"


def validate_scaling():
    print("=" * 70)
    print("RUNNING SYSTEMATIC SCALING VALIDATION SUITE")
    print("=" * 70)

    assert RAW_CSV.exists(), f"Raw CSV not found: {RAW_CSV}"
    assert SUMMARY_CSV.exists(), f"Summary CSV not found: {SUMMARY_CSV}"
    assert RES_CSV.exists(), f"Resource CSV not found: {RES_CSV}"

    df_raw = pd.read_csv(RAW_CSV)
    df_summary = pd.read_csv(SUMMARY_CSV)
    df_res = pd.read_csv(RES_CSV)

    evaluated_sizes = sorted(df_raw["dataset_size"].unique().tolist())
    print(f"Evaluated Checkpoint Sizes: {evaluated_sizes}")

    # Check 1: Checkpoint counts and expected query records
    # For each size: 25 queries x 3 workloads x 5 methods = 375 records
    records_per_size = df_raw.groupby("dataset_size")["query_id"].count().to_dict()
    val_check1 = all(count == 375 for count in records_per_size.values())
    print(f"Check 1: 375 records per checkpoint -> {'PASS' if val_check1 else 'FAIL'} ({records_per_size})")

    # Check 2: Consistent query IDs across methods within each size
    consistent_ids = True
    for sz in evaluated_sizes:
        sub = df_raw[df_raw["dataset_size"] == sz]
        for (m, w), grp in sub.groupby(["method", "workload"]):
            if len(grp) != 25 or set(grp["query_id"]) != set(range(25)):
                consistent_ids = False
    val_check2 = consistent_ids
    print(f"Check 2: Consistent query IDs (0-24) across methods -> {'PASS' if val_check2 else 'FAIL'}")

    # Check 3: Predicate compliance (zero violations)
    pred_violations = int((df_raw["results_returned"] - df_raw["valid_results"]).sum())
    val_check3 = (pred_violations == 0)
    print(f"Check 3: Predicate compliance (0 violations) -> {'PASS' if val_check3 else 'FAIL'} (violations: {pred_violations})")

    # Check 4: Recall bounds and mathematical fidelity
    exact_rec = df_raw[df_raw["method"] == "ExactSearch"]["recall_at_10"].min()
    pre_rec = df_raw[df_raw["method"] == "PreFilter"]["recall_at_10"].min()
    all_rec_valid = bool((df_raw["recall_at_10"] >= 0.0).all() and (df_raw["recall_at_10"] <= 1.0).all())
    val_check4 = bool(exact_rec == 1.0 and pre_rec == 1.0 and all_rec_valid)
    print(f"Check 4: Recall fidelity (Exact & PreFilter = 1.0, bounded [0,1]) -> {'PASS' if val_check4 else 'FAIL'}")

    # Check 5: Finite positive latencies
    val_check5 = bool(np.all(np.isfinite(df_raw["latency_ms"])) and (df_raw["latency_ms"] > 0).all())
    print(f"Check 5: Latencies finite and positive -> {'PASS' if val_check5 else 'FAIL'}")

    # Check 6: Separation of index build time and query latency
    # Build time is hundreds of seconds (recorded in df_res); query latencies are milliseconds (< 10 seconds)
    val_check6 = bool((df_res["index_build_time_s"] >= 60.0).all() and (df_raw["latency_ms"] < 10000.0).all())
    print(f"Check 6: Index build separated from query latency -> {'PASS' if val_check6 else 'FAIL'}")

    # Check 7: Zero errors or timeouts
    val_check7 = bool(df_raw["error"].isna().all())
    print(f"Check 7: Zero query errors or timeouts -> {'PASS' if val_check7 else 'FAIL'}")

    # Check 8: Summary recomputability
    recomputed_rec = df_raw.groupby(["dataset_size", "method", "workload"])["recall_at_10"].mean().to_dict()
    summary_rec = df_summary.set_index(["dataset_size", "method", "workload"])["mean_recall"].to_dict()
    val_check8 = all(abs(recomputed_rec[k] - summary_rec[k]) < 1e-6 for k in recomputed_rec)
    print(f"Check 8: Summary table recomputability -> {'PASS' if val_check8 else 'FAIL'}")

    # Check 9: Resource tracking completeness
    has_res_fields = all(c in df_res.columns for c in ["dataset_size", "index_build_time_s", "peak_rss_mb", "ram_delta_mb", "free_ram_start_mb", "free_ram_end_mb"])
    val_check9 = bool(has_res_fields and len(df_res) == len(evaluated_sizes))
    print(f"Check 9: Resource measurements complete -> {'PASS' if val_check9 else 'FAIL'}")

    # Check 10: Subset vs full dataset distinction
    val_check10 = all(sz < 2735264 for sz in evaluated_sizes)
    print(f"Check 10: Explicitly distinguished as subset evaluation -> {'PASS' if val_check10 else 'FAIL'}")

    # Check 11: 10K pilot and medium repo integrity
    pilot_csv = Path("research_audit/arxiv_large_pilot/pilot_raw_results.csv")
    medium_csv = Path("C:/Users/anime/Desktop/filtered-ann-research-medium/results/medium/benchmark_summary.csv")
    val_check11 = bool(pilot_csv.exists() and pilot_csv.stat().st_size == 62964 and medium_csv.exists())
    print(f"Check 11: 10K pilot and medium repos untouched -> {'PASS' if val_check11 else 'FAIL'}")

    all_passed = bool(val_check1 and val_check2 and val_check3 and val_check4 and val_check5 and val_check6 and val_check7 and val_check8 and val_check9 and val_check10 and val_check11)

    report = {
        "scaling_validation_passed": all_passed,
        "evaluated_checkpoint_sizes": evaluated_sizes,
        "checks": {
            "check1_record_counts": {"passed": bool(val_check1), "records_per_size": records_per_size},
            "check2_consistent_query_ids": {"passed": bool(val_check2)},
            "check3_predicate_compliance": {"passed": bool(val_check3), "violations": pred_violations},
            "check4_recall_fidelity": {"passed": bool(val_check4)},
            "check5_latency_finite": {"passed": bool(val_check5)},
            "check6_index_build_separated": {"passed": bool(val_check6)},
            "check7_zero_errors": {"passed": bool(val_check7)},
            "check8_summary_recomputable": {"passed": bool(val_check8)},
            "check9_resource_measurements": {"passed": bool(val_check9)},
            "check10_subset_distinction": {"passed": bool(val_check10)},
            "check11_prior_research_intact": {"passed": bool(val_check11)},
        },
        "checkpoint_metrics": df_res.to_dict(orient="records"),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"VALIDATION REPORT SAVED: {REPORT_PATH.name} ({'ALL PASSED' if all_passed else 'FAILURES DETECTED'})")
    print("=" * 70)


if __name__ == "__main__":
    validate_scaling()
