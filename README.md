# Filtered Approximate Nearest Neighbor Search (FANNS) on arXiv-Large

Empirical scaling study and systems feasibility analysis for metadata-filtered Approximate Nearest Neighbor Search (FANNS) using the official `SPCL/arxiv-for-fanns-large` dataset (Iff et al., 2025).

---

## 1. Project Overview

This repository investigates the empirical recall-latency trade-offs and physical resource scaling behavior of metadata-filtered vector search on high-dimensional scientific paper embeddings:
- **Embedding Dimensionality**: $D = 4,096$
- **Data Type**: `float32` unit-normalized (16,388 bytes per vector record)
- **Workload Filters**:
  - `EM`: Exact metadata equality matching (single category)
  - `Range`: Numeric interval range filtering (update dates / citation ranges)
  - `EMIS`: Conjunctive multi-attribute filtering (category equality $\land$ date range)
- **Evaluated Algorithms**:
  - `ExactSearch`: Brute-force linear scan with exact metadata filtering (Ground Truth baseline)
  - `PreFilter`: Metadata filtering followed by vectorized BLAS distance ranking
  - `HNSW_PostFilter`: Unconstrained HNSW graph exploration followed by metadata filtering
  - `InGraphFilter`: Metadata-constrained traversal during HNSW beam search
  - `AdaptiveRouter_2pct`: Dynamic query routing based on estimated selectivity threshold ($\sigma \le 2\%$)

---

## 2. Experimental Progression & Key Findings

### Completed & Validated Checkpoints (10K to 100K)
The study successfully completed and validated 1,500 method-query observations across four checkpoints:
1. **10,000 vectors** (Pilot reference benchmark: 118.3 s build, 294.9 MB peak RSS)
2. **25,000 vectors** (Scaling Checkpoint 1: 242.8 s build, 458.1 MB peak RSS)
3. **50,000 vectors** (Scaling Checkpoint 2: 547.0 s build, 1,023.2 MB peak RSS)
4. **100,000 vectors** (Scaling Checkpoint 3: 670.1 s build, 1,720.0 MB peak RSS, 514 MB available physical RAM)

### Core Scientific Findings:
- **Catastrophic Post-Filtering Collapse**: Unconstrained HNSW search followed by metadata filtering suffers severe candidate starvation on high-dimensional embeddings ($0.05$ to $0.17$ Recall@10 on Range queries at 50K/100K).
- **InGraph Recall Invariance**: In-graph traversal maintains high Recall@10 ($0.92$ to $0.98$) across scale.
- **Topological Wandering Resolved**: On sparse queries ($\sigma \le 2\%$), InGraph search wanders through thousands of unindexed nodes (taking up to 1,277 ms per query). The **Adaptive Router (2% threshold)** dynamically routes sparse queries to PreFilter, reducing Range workload latency by **$2.4\times$ to $3.0\times$** down to **sub-48ms** while maintaining $>93\%$ recall.
- **500K / 1M Systems Gating**: 4,096-dim float32 vectors require 16.4 MB/1K vectors. At 500K vectors, the raw dataset (8.2 GB) and graph memory (7.3 GB) exceed host physical RAM (7.33 GB), forcing $>5.5\text{ GB}$ into SSD pagefile thrashing ($1,000\times$ latency penalty). The 500K checkpoint was formally **GATED** under the safety policy, and 1M was declared **DO NOT RUN LOCALLY (HPC NODE REQUIRED)**.

---

## 3. Repository Architecture & Key Artifacts

```
filtered-ann-research-large/
├── README.md
├── .gitignore
├── download_pilot_data.py               # Pilot dataset extraction
├── prepare_scaling_data.py              # Incremental dataset fetcher (HTTP range requests)
├── run_large_pilot.py                   # 10K pilot execution pipeline
├── run_scaling_checkpoint.py            # 25K and 50K execution pipeline
├── run_scaling_100k.py                  # 100K execution and multi-checkpoint consolidation
├── validate_scaling_results.py          # 11-gate scaling validation suite
├── validate_scaling_updated.py          # 12-gate updated validation suite
├── generate_plots_updated.py            # Figure generation for 10K-100K study
├── generate_500k_figures.py             # Figure generation for 500K feasibility study
└── research_audit/
    ├── arxiv_large_pilot/               # 10K Pilot results, figures, and logs
    └── arxiv_large_scaling/             # Multi-checkpoint scaling audit (10K, 25K, 50K, 100K)
        ├── SCALING_PLAN_100K_500K_1M.md
        ├── SCALING_FINDINGS_UPDATED.md  # Comprehensive multi-scale report
        ├── FEASIBILITY_500K_1M.md       # Systems capacity modeling & HPC roadmap
        ├── scaling_raw_results_updated.csv   # 1,500 validated method-query rows
        ├── scaling_summary_updated.csv       # Summary metrics (Recall, Latency, QPS)
        ├── resource_measurements_updated.csv # Measured build times and peak RSS
        ├── validation_report_updated.json    # 12/12 validation report
        ├── figures/
        │   ├── figure1_resource_scaling_updated.png
        │   └── figure2_recall_latency_scaling_updated.png
        └── 500k/                        # 500K Gating & Preflight Audit Artifacts
            ├── 500k_run_plan.md
            ├── 500k_findings.md
            ├── 500k_resource_measurements.csv
            ├── 500k_raw_results.csv
            ├── 500k_summary.csv
            ├── 500k_validation_report.json
            ├── 500k_run_log.txt
            └── figures/
                ├── 500k_resource_scaling.png
                └── 500k_recall_latency.png
```

---

## 4. Hardware Environment & Validation

- **Processor**: AMD Ryzen 3 5300U (4 Cores / 8 Threads @ 2.60 GHz)
- **Memory**: 8 GB DDR4 (7.33 GB visible / 7,502 MB), 8 GB Pagefile
- **Storage**: 512 GB PCIe NVMe SSD (>270 GB free)
- **Validation**: 100% passed across all 12 validation gates in `validation_report_updated.json`.
