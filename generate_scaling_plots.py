"""
Generate Presentation Plots for arXiv-Large Scaling Results.
Figure 1: Resource Scaling (Build Time & Peak RSS vs Physical Limits).
Figure 2: Quality & Latency Scaling (Recall@10 and Latency vs Database Size).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT_DIR = Path("research_audit/arxiv_large_scaling")
FIGURES_DIR = AUDIT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Combine 10K pilot data with 25K and 50K scaling data for complete visualization
PILOT_RES = Path("research_audit/arxiv_large_pilot/pilot_validation.json")
SCALING_RES = AUDIT_DIR / "resource_measurements.csv"
SCALING_SUMMARY = AUDIT_DIR / "scaling_summary.csv"
PILOT_SUMMARY = Path("research_audit/arxiv_large_pilot/pilot_summary.csv")


def generate_plots():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Load resources
    df_res = pd.read_csv(SCALING_RES)
    # Add 10K point
    res_10k = {
        "dataset_size": 10000,
        "raw_vector_file_mb": 163.88,
        "index_build_time_s": 118.33,
        "index_file_size_mb": 157.32,
        "peak_rss_mb": 294.93,
    }
    df_all_res = pd.concat([pd.DataFrame([res_10k]), df_res], ignore_index=True).sort_values("dataset_size")

    # -------------------------------------------------------------
    # FIGURE 1: RESOURCE SCALING & HARDWARE CEILINGS
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Index Build Time
    sizes_k = df_all_res["dataset_size"] / 1000.0
    build_times_min = df_all_res["index_build_time_s"] / 60.0

    ax1.plot(sizes_k, build_times_min, marker='o', color='#1f77b4', linewidth=2.2, markersize=8, label="Measured Build Time")
    for x, y in zip(sizes_k, build_times_min):
        ax1.annotate(f"{y:.2f} min\n({y*60:.0f}s)", (x, y + 0.3), fontsize=9, ha='center')

    # Extrapolations for 100K, 250K, 500K
    extrap_sizes = np.array([50, 100, 250, 500])
    # T(N) = c * N * log(N)
    c_fit = df_all_res["index_build_time_s"].iloc[-1] / (50000 * np.log(50000))
    extrap_times_min = (c_fit * (extrap_sizes * 1000) * np.log(extrap_sizes * 1000)) / 60.0
    ax1.plot(extrap_sizes, extrap_times_min, linestyle='--', color='#7f7f7f', alpha=0.8, label="Projected O(N log N) Build Time")

    ax1.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Index Construction Time (Minutes)", fontsize=11, fontweight="bold")
    ax1.set_title("HNSW Index Build Time Scaling (M=16, efC=50)", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 25)
    ax1.legend(loc="upper left", frameon=True)

    # Subplot 2: Peak RSS vs Hardware Limits
    peak_rss_gb = df_all_res["peak_rss_mb"] / 1024.0
    raw_file_gb = df_all_res["raw_vector_file_mb"] / 1024.0

    ax2.plot(sizes_k, peak_rss_gb, marker='s', color='#d62728', linewidth=2.2, markersize=8, label="Measured Process Peak RSS")
    ax2.plot(sizes_k, raw_file_gb, marker='^', color='#2ca02c', linestyle=':', linewidth=1.8, markersize=7, label="Raw Vector File Size (Disk)")

    # Projected RSS
    slope_rss = (df_all_res["peak_rss_mb"].iloc[-1] - df_all_res["peak_rss_mb"].iloc[0]) / (50000 - 10000)
    extrap_rss_gb = (df_all_res["peak_rss_mb"].iloc[0] + (extrap_sizes * 1000 - 10000) * slope_rss) / 1024.0
    extrap_raw_gb = (extrap_sizes * 1000 * 16388) / (1024.0 * 1024.0 * 1024.0)

    ax2.plot(extrap_sizes, extrap_rss_gb, linestyle='--', color='#d62728', alpha=0.5, label="Projected Peak RSS")
    ax2.plot(extrap_sizes, extrap_raw_gb, linestyle='--', color='#2ca02c', alpha=0.5, label="Projected Raw Vectors (Disk)")

    # Hardware Limits
    ax2.axhline(7.33, color='black', linestyle='-', linewidth=2.0, label="Total Laptop RAM (7.33 GB Visible)")
    ax2.axhline(1.50, color='orange', linestyle='--', linewidth=1.5, label="Available Free RAM (~1.5 GB)")
    ax2.fill_between([0, 120], 7.33, 20, color='red', alpha=0.12, label="Hardware OOM / Crash Zone")

    ax2.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Memory Consumption (Gigabytes)", fontsize=11, fontweight="bold")
    ax2.set_title("Memory Scaling vs. Laptop Hardware Ceilings", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 10.0)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    FIG1_PATH = FIGURES_DIR / "figure1_resource_scaling.png"
    fig.savefig(FIG1_PATH)
    plt.close(fig)
    print(f"[SAVED] {FIG1_PATH.name}")

    # -------------------------------------------------------------
    # FIGURE 2: RECALL & LATENCY DYNAMICS ACROSS SIZES
    # -------------------------------------------------------------
    df_sum_scaling = pd.read_csv(SCALING_SUMMARY)
    df_sum_pilot = pd.read_csv(PILOT_SUMMARY)
    df_sum_pilot["dataset_size"] = 10000
    df_sum_all = pd.concat([df_sum_pilot, df_sum_scaling], ignore_index=True)

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    methods = ["HNSW_PostFilter", "InGraphFilter", "AdaptiveRouter_2pct", "PreFilter"]
    colors = {"HNSW_PostFilter": "#d62728", "InGraphFilter": "#2ca02c", "AdaptiveRouter_2pct": "#1f77b4", "PreFilter": "#ff7f0e"}
    markers = {"HNSW_PostFilter": "o", "InGraphFilter": "s", "AdaptiveRouter_2pct": "D", "PreFilter": "^"}

    # Subplot 1: Recall@10 across Dataset Sizes (Range Workload)
    for m in methods:
        sub = df_sum_all[(df_sum_all["method"] == m) & (df_sum_all["workload"] == "Range")].sort_values("dataset_size")
        ax3.plot(sub["dataset_size"]/1000.0, sub["mean_recall"], marker=markers[m], color=colors[m], label=m, linewidth=2.0, markersize=8)

    ax3.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean Recall@10 (Range Workload)", fontsize=11, fontweight="bold")
    ax3.set_title("Recall Stability Across Scale (Range Workload)", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks([10, 25, 50])
    ax3.legend(loc="lower left", frameon=True)

    # Subplot 2: Latency across Dataset Sizes (Range Workload)
    for m in methods:
        sub = df_sum_all[(df_sum_all["method"] == m) & (df_sum_all["workload"] == "Range")].sort_values("dataset_size")
        ax4.plot(sub["dataset_size"]/1000.0, sub["mean_latency_ms"], marker=markers[m], color=colors[m], label=f"{m} (Mean)", linewidth=2.0, markersize=8)

    ax4.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Mean Latency (ms)", fontsize=11, fontweight="bold")
    ax4.set_title("Query Latency Growth (Range Workload)", fontsize=12, fontweight="bold")
    ax4.set_xticks([10, 25, 50])
    ax4.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    FIG2_PATH = FIGURES_DIR / "figure2_recall_latency_scaling.png"
    fig.savefig(FIG2_PATH)
    plt.close(fig)
    print(f"[SAVED] {FIG2_PATH.name}")


if __name__ == "__main__":
    generate_plots()
