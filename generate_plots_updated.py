"""
Figure Generation Script for arXiv-Large Multi-Checkpoint Scaling Study (10K, 25K, 50K, 100K).
Generates publication-quality figures:
- figure1_resource_scaling_updated.png: Construction time, memory footprint, and hardware safety limits.
- figure2_recall_latency_scaling_updated.png: Recall fidelity, latency scaling curves, and router dispatch dynamics.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT_DIR = Path("research_audit/arxiv_large_scaling")
FIGURES_DIR = AUDIT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = AUDIT_DIR / "scaling_raw_results_updated.csv"
SUMMARY_CSV = AUDIT_DIR / "scaling_summary_updated.csv"
RES_CSV = AUDIT_DIR / "resource_measurements_updated.csv"


def plot_resource_scaling():
    df_res = pd.read_csv(RES_CSV).sort_values("dataset_size")
    sizes_k = df_res["dataset_size"].values / 1000.0
    build_times_min = df_res["index_build_time_s"].values / 60.0
    peak_rss_mb = df_res["peak_rss_mb"].values

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Index Build Time
    ax1.plot(sizes_k, build_times_min, marker="o", markersize=8, color="#1f77b4", linewidth=2.5, label="Measured Build Time")
    for x, y in zip(sizes_k, build_times_min):
        ax1.annotate(f"{y:.1f} min\n({int(x)}K)", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color="#1f77b4")

    # Extrapolation curve to 500K and 1M
    extrap_sizes_k = np.array([10, 25, 50, 100, 250, 500, 1000])
    # O(N log N) scaling model calibrated on 10K-100K
    c_time = (build_times_min[-1] * 60.0) / (100000 * np.log2(100000))
    proj_time_min = (c_time * (extrap_sizes_k * 1000) * np.log2(extrap_sizes_k * 1000)) / 60.0

    ax1.plot(extrap_sizes_k[3:], proj_time_min[3:], linestyle="--", color="#e377c2", alpha=0.8, label=f"Projected O(N log N) Curve")
    ax1.scatter([500, 1000], [proj_time_min[5], proj_time_min[6]], color="#d62728", marker="x", s=80, zorder=5, label="500K / 1M Extrapolations (Gated)")
    ax1.annotate(f"Proj: {proj_time_min[5]:.1f} min\n(~{proj_time_min[5]/60:.1f} hrs)", (500, proj_time_min[5]), textcoords="offset points", xytext=(-20, 10), fontsize=8.5, color="#d62728")
    ax1.annotate(f"Proj: {proj_time_min[6]:.1f} min\n(~{proj_time_min[6]/60:.1f} hrs)", (1000, proj_time_min[6]), textcoords="offset points", xytext=(-30, 10), fontsize=8.5, color="#d62728")

    ax1.set_title("HNSW Index Construction Time vs Dataset Size\n(M=16, efC=50, D=4096)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Construction Time (Minutes)", fontsize=11, fontweight="semibold")
    ax1.set_xlim(0, 1050)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Peak Memory Footprint & Hardware Ceilings
    ax2.plot(sizes_k, peak_rss_mb, marker="s", markersize=8, color="#2ca02c", linewidth=2.5, label="Measured Peak RSS")
    for x, y in zip(sizes_k, peak_rss_mb):
        ax2.annotate(f"{y:.0f} MB", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color="#2ca02c")

    # Extrapolation for memory
    # Slope based on 10K to 100K delta: (peak_rss[-1] - peak_rss[0]) / 90
    rss_slope = (peak_rss_mb[-1] - peak_rss_mb[0]) / 90.0
    proj_rss_mb = peak_rss_mb[0] + rss_slope * (extrap_sizes_k - 10)

    ax2.plot(extrap_sizes_k[3:], proj_rss_mb[3:], linestyle="--", color="#ff7f0e", alpha=0.8, label="Projected Linear RSS Growth")
    ax2.scatter([500, 1000], [proj_rss_mb[5], proj_rss_mb[6]], color="#d62728", marker="x", s=80, zorder=5, label="500K / 1M Extrapolations")

    # Hardware Limits
    ax2.axhline(7502, color="#d62728", linestyle="-.", linewidth=2, label="Laptop Visible Physical RAM (7.33 GB)")
    ax2.axhline(15694, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Total Virtual Memory Limit (RAM + Pagefile)")

    ax2.annotate("500K Proj: ~9.2 GB\n(EXCEEDS PHYSICAL RAM)", (500, proj_rss_mb[5]), textcoords="offset points", xytext=(-60, 15), fontsize=8.5, fontweight="bold", color="#d62728")
    ax2.annotate("1M Proj: ~18.5 GB\n(EXCEEDS VIRTUAL MEMORY)", (1000, proj_rss_mb[6]), textcoords="offset points", xytext=(-80, -25), fontsize=8.5, fontweight="bold", color="#d62728")

    ax2.set_title("Peak Resident Memory (RSS) vs Hardware Ceilings\n(AMD Ryzen 3 5300U, 8GB RAM)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Resident Memory (MB)", fontsize=11, fontweight="semibold")
    ax2.set_xlim(0, 1050)
    ax2.set_ylim(0, 20000)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure1_resource_scaling_updated.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE 1 SAVED] {out_path.name}")


def plot_recall_latency_scaling():
    df_raw = pd.read_csv(RAW_CSV)
    df_summary = pd.read_csv(SUMMARY_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    (ax1, ax2, ax3) = axes

    methods = ["ExactSearch", "InGraphFilter", "PreFilter", "AdaptiveRouter_2pct", "HNSW_PostFilter"]
    palette = {
        "ExactSearch": "#333333",
        "InGraphFilter": "#1f77b4",
        "PreFilter": "#ff7f0e",
        "AdaptiveRouter_2pct": "#2ca02c",
        "HNSW_PostFilter": "#d62728",
    }
    markers = {
        "ExactSearch": "o",
        "InGraphFilter": "^",
        "PreFilter": "s",
        "AdaptiveRouter_2pct": "D",
        "HNSW_PostFilter": "X",
    }

    # 1. Mean Recall@10 across scale
    for m in methods:
        sub = df_summary[df_summary["method"] == m]
        agg = sub.groupby("dataset_size")["mean_recall"].mean().reset_index().sort_values("dataset_size")
        sizes_k = agg["dataset_size"] / 1000.0
        ax1.plot(sizes_k, agg["mean_recall"], marker=markers[m], color=palette[m], label=m, linewidth=2.2, markersize=7)

    ax1.set_title("Recall@10 Stability Across Scale\n(Evaluated Active Prefix Ground Truth)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=10.5, fontweight="semibold")
    ax1.set_ylabel("Mean Recall@10", fontsize=10.5, fontweight="semibold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks([10, 25, 50, 100])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True, fontsize=9)

    # 2. Overall Mean Query Latency (log scale)
    for m in methods:
        sub = df_summary[df_summary["method"] == m]
        agg = sub.groupby("dataset_size")["mean_latency_ms"].mean().reset_index().sort_values("dataset_size")
        sizes_k = agg["dataset_size"] / 1000.0
        ax2.plot(sizes_k, agg["mean_latency_ms"], marker=markers[m], color=palette[m], label=m, linewidth=2.2, markersize=7)

    ax2.set_yscale("log")
    ax2.set_title("Overall Mean Query Latency vs Dataset Size\n(All 3 Workloads Averaged, Log Scale)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=10.5, fontweight="semibold")
    ax2.set_ylabel("Mean Latency (ms, Log Scale)", fontsize=10.5, fontweight="semibold")
    ax2.set_xticks([10, 25, 50, 100])
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    # 3. Range Workload Latency Scaling (Highlighting Router Benefit)
    range_summary = df_summary[df_summary["workload"] == "Range"]
    for m in methods:
        sub = range_summary[range_summary["method"] == m].sort_values("dataset_size")
        sizes_k = sub["dataset_size"] / 1000.0
        ax3.plot(sizes_k, sub["mean_latency_ms"], marker=markers[m], color=palette[m], label=m, linewidth=2.2, markersize=7)

    ax3.set_title("Range Workload Latency Scaling\n(Dynamic Router Offloading Sparse Queries)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=10.5, fontweight="semibold")
    ax3.set_ylabel("Mean Latency (ms)", fontsize=10.5, fontweight="semibold")
    ax3.set_xticks([10, 25, 50, 100])
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure2_recall_latency_scaling_updated.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE 2 SAVED] {out_path.name}")


if __name__ == "__main__":
    plot_resource_scaling()
    plot_recall_latency_scaling()
