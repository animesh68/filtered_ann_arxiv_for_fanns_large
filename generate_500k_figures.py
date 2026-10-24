"""
Script to generate publication-quality figures for the 500K scaling feasibility study:
- 500k_resource_scaling.png: Construction time and memory footprint vs host physical and commit limits.
- 500k_recall_latency.png: Recall stability, query latency scaling, and adaptive router dynamics across scale.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT_DIR = Path("research_audit/arxiv_large_scaling")
OUT_DIR = AUDIT_DIR / "500k" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RES_CSV = AUDIT_DIR / "resource_measurements_updated.csv"
SUMMARY_CSV = AUDIT_DIR / "scaling_summary_updated.csv"


def plot_500k_resource_scaling():
    df_res = pd.read_csv(RES_CSV).sort_values("dataset_size")
    sizes_k = df_res["dataset_size"].values / 1000.0
    build_times_min = df_res["index_build_time_s"].values / 60.0
    peak_rss_mb = df_res["peak_rss_mb"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Construction Time
    ax1.plot(sizes_k, build_times_min, marker="o", markersize=8, color="#1f77b4", linewidth=2.5, label="Measured Build Time")
    for x, y in zip(sizes_k, build_times_min):
        ax1.annotate(f"{y:.1f} min\n({int(x)}K)", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color="#1f77b4")

    # Extrapolation curve to 500K
    extrap_k = np.array([10, 25, 50, 100, 200, 300, 400, 500])
    c_time = (build_times_min[-1] * 60.0) / (100000 * np.log2(100000))
    proj_time_min = (c_time * (extrap_k * 1000) * np.log2(extrap_k * 1000)) / 60.0

    ax1.plot(extrap_k[3:], proj_time_min[3:], linestyle="--", color="#e377c2", alpha=0.8, label="Projected O(N log N) Curve (Clean DRAM)")
    ax1.scatter([500], [proj_time_min[-1]], color="#d62728", marker="x", s=100, zorder=5, label="500K Clean Projection (~63.7 min)")
    ax1.annotate("500K Clean Proj: 63.7 min\n(Thrashing Est: >15-20 hrs)", (500, proj_time_min[-1]), textcoords="offset points", xytext=(-80, 15), fontsize=8.5, fontweight="bold", color="#d62728")

    ax1.set_title("HNSW Index Construction Time vs Dataset Size\n(M=16, efC=50, D=4096)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Construction Time (Minutes)", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 550)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Peak Memory Footprint & Safety Gate
    ax2.plot(sizes_k, peak_rss_mb, marker="s", markersize=8, color="#2ca02c", linewidth=2.5, label="Measured Peak RSS")
    for x, y in zip(sizes_k, peak_rss_mb):
        ax2.annotate(f"{y:.0f} MB", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color="#2ca02c")

    # Linear extrapolation based on 50K->100K marginal RSS (13.94 KB/vec)
    proj_500k_rss = peak_rss_mb[-1] + (400000 * 0.013936)
    ax2.plot([100, 500], [peak_rss_mb[-1], proj_500k_rss], linestyle="--", color="#ff7f0e", alpha=0.8, label="Projected Linear RSS Growth")
    ax2.scatter([500], [proj_500k_rss], color="#d62728", marker="x", s=100, zorder=5, label="500K Projected RSS (7,294 MB)")

    # Ceilings
    ax2.axhline(7502, color="#d62728", linestyle="-.", linewidth=2, label="Laptop Visible Physical RAM (7.33 GB / 7,502 MB)")
    ax2.axhline(15694, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Total Virtual Memory Commit Limit (15.33 GB)")
    ax2.axhspan(7502, 15694, color="#d62728", alpha=0.08, label="Heavy Paging & Thrashing Zone")

    ax2.annotate("500K Projected RSS: 7,294 MB\n(97.2% of Physical RAM;\nExceeds 1.75 GB Free RAM)", (500, proj_500k_rss), textcoords="offset points", xytext=(-160, -35), fontsize=8.5, fontweight="bold", color="#d62728")

    ax2.set_title("Peak Resident Memory (RSS) vs Hardware Safety Ceilings\n(AMD Ryzen 3 5300U, 8GB RAM)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Resident Memory (MB)", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 550)
    ax2.set_ylim(0, 18000)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_file = OUT_DIR / "500k_resource_scaling.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[FIGURE SAVED] {out_file}")


def plot_500k_recall_latency():
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

    # 1. Recall stability
    for m in methods:
        sub = df_summary[df_summary["method"] == m]
        agg = sub.groupby("dataset_size")["mean_recall"].mean().reset_index().sort_values("dataset_size")
        sizes_k = agg["dataset_size"] / 1000.0
        ax1.plot(sizes_k, agg["mean_recall"], marker=markers[m], color=palette[m], label=m, linewidth=2.2, markersize=7)

    ax1.set_title("Recall@10 Stability Across Scale\n(Evaluated Active Prefix Ground Truth)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Mean Recall@10", fontsize=10.5, fontweight="bold")
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
    ax2.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Mean Latency (ms, Log Scale)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks([10, 25, 50, 100])
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    # 3. Range Workload Latency Scaling
    range_summary = df_summary[df_summary["workload"] == "Range"]
    for m in methods:
        sub = range_summary[range_summary["method"] == m].sort_values("dataset_size")
        sizes_k = sub["dataset_size"] / 1000.0
        ax3.plot(sizes_k, sub["mean_latency_ms"], marker=markers[m], color=palette[m], label=m, linewidth=2.2, markersize=7)

    ax3.set_title("Range Workload Latency Scaling\n(Dynamic Router Offloading Sparse Queries)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Dataset Size (Thousands of Vectors)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Mean Latency (ms)", fontsize=10.5, fontweight="bold")
    ax3.set_xticks([10, 25, 50, 100])
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    out_file = OUT_DIR / "500k_recall_latency.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[FIGURE SAVED] {out_file}")


if __name__ == "__main__":
    plot_500k_resource_scaling()
    plot_500k_recall_latency()
