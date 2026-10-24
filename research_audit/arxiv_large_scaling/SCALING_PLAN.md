# arXiv-Large Scaling Plan & Resource Feasibility Assessment

**Author**: Senior Systems Research Engineer  
**Date**: 2026-10-24  
**Project**: Filtered Approximate Nearest Neighbor Search (FANNS)  
**Dataset**: `SPCL/arxiv-for-fanns-large` (Iff et al., 2025)  
**Directory**: `research_audit/arxiv_large_scaling/`  
**Prerequisite Baseline**: `research_audit/arxiv_large_pilot/` (Verified Intact)

---

## 1. Current Measured Hardware & Environment State

Measurements verified via Windows CIM/WMI queries immediately prior to scaling execution:

| Parameter | Actual Measured Value |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home (Build 26100) |
| **CPU Model** | AMD Ryzen 3 5300U with Radeon Graphics |
| **Cores / Logical Processors** | 4 Physical Cores / 8 Logical Processors |
| **Total Visible RAM** | 7,682,164 KB (**7.33 GB**) |
| **Available Free RAM** | 1,183,348 KB (**1.13 GB free** at start of scaling inspection) |
| **Drive C: Total Capacity** | 510,825,328,640 bytes (**475.74 GB**) |
| **Drive C: Free Space** | 302,211,002,368 bytes (**281.46 GB free**) |
| **Python Binary** | Python 3.14.6 (`C:\Users\anime\Desktop\filtered-ann-research-small\venv`) |
| **Core Libraries** | `numpy==2.5.3`, `pandas==3.0.6`, `scipy==1.18.1`, `psutil==7.2.2`, `matplotlib==3.11.2`, `tqdm==4.70.1` |
| **Memory-Mapped I/O** | `numpy.memmap` validated in 10K pilot (zero full-array RAM copies) |

---

## 2. Baseline Verification: 10K Reference Pilot

The original 10,000-vector pilot artifacts in `research_audit/arxiv_large_pilot/` were verified completely intact:
- `hnsw_pilot_index.pkl`: 164.96 MB (Enter point = 2902, Max layer = 4, Nodes = 10,000)
- `pilot_raw_results.csv`: 750 records across 5 methods $\times$ 3 workloads $\times$ 50 queries
- `pilot_summary.csv`: Recomputable from raw records with zero discrepancies
- `pilot_validation.json`: 10/10 systematic validation checks PASSED
- `ENVIRONMENT_AND_FEASIBILITY.md` & `PILOT_FINDINGS.md`: Intact and preserved

### Baseline 10K Performance Metrics
- **Dataset Size**: $N = 10,000$, $D = 4,096$, float32 (163.88 MB on disk)
- **Index Build Time**: 118.33 seconds (1.97 minutes)
- **Peak Process RSS**: 294.9 MB (RAM delta = 156.2 MB over base Python)
- **Query Evaluation Time**: 24.69 seconds for 750 queries (30.4 QPS overall)
- **Observed Behavior**:
  - `HNSW_PostFilter`: Severe recall collapse (38.2% on EM, 28.6% on EMIS, 6.0% on Range)
  - `InGraphFilter`: Preserves recall (95.0% to 97.6%) with P95 latency tail (345.8 ms on Range)
  - `PreFilter`: 100% recall, latency directly proportional to candidate set size
  - `AdaptiveRouter_2pct`: $3.6\times$ P95 latency reduction on Range (95.7 ms vs 345.8 ms) while maintaining 96.8% recall

---

## 3. Scaling Memory & Storage Sizing Analysis

Raw vector storage for $N$ vectors ($D=4,096$, float32, 16,388 bytes/record):
$$B_{\text{raw}} = N \times (4 + 4,096 \times 4)\text{ bytes}$$

| Checkpoint | Vectors ($N$) | Raw Disk File | In-Memory Graph RAM | Estimated Peak RSS | Safety Evaluation (on 7.33 GB Laptop / 1.13 GB Free) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **10K** | 10,000 | 163.88 MB | ~12 MB | **295 MB** | **Reference Complete** (Observed) |
| **25K** | 25,000 | 409.70 MB | ~30 MB | **~520 MB** | **SAFE**: High confidence; RAM delta $<350$ MB |
| **50K** | 50,000 | 819.40 MB | ~60 MB | **~900 MB** | **SAFE**: Feasible within free RAM buffer |
| **100K** | 100,000 | 1,638.80 MB | ~120 MB | **~1.76 GB** | **BORDERLINE**: Exceeds immediate free RAM; triggers Windows paging |
| **250K** | 250,000 | 4,097.00 MB | ~300 MB | **~4.20 GB** | **HIGH RISK**: Consumes $>57\%$ of total system RAM; heavy paging |
| **500K** | 500,000 | 8,194.00 MB | ~600 MB | **$>8.50$ GB** | **UNSAFE / NO-GO**: Raw file alone (8.19 GB) exceeds laptop visible RAM (7.33 GB) |
| **1M** | 1,000,000 | 16,388.00 MB | ~1.20 GB | **$>18.0$ GB** | **IMPOSSIBLE on Laptop**: Requires HPC Node (64GB+ RAM) |

---

## 4. Execution Checkpoint Sequence & Safety Gates

We will scale incrementally through the following checkpoints:

1. **Checkpoint 1 (10K)**: Validated baseline reference.
2. **Checkpoint 2 (25K)**:
   - Stream additional 15,000 vectors from `SPCL/arxiv-for-fanns-large` to form the 25,000-vector prefix.
   - Construct HNSW index ($M=16, ef_{\text{construction}}=50$).
   - Run 5-query smoke test + 25 queries per workload (75 queries $\times$ 4 core methods = 300 evaluations).
   - Record actual peak RSS, build time, and latency.
3. **Checkpoint 3 (50K)**:
   - If Checkpoint 2 peak RSS $\le 750$ MB and build time $\le 6$ minutes: proceed to 50,000 vectors.
   - Run benchmark on identical fixed queries.
   - Measure actual memory growth slope $\frac{\Delta \text{RSS}}{\Delta N}$.
4. **Checkpoint 4 (100K)**:
   - If Checkpoint 3 passes safely and total system responsiveness is maintained: evaluate 100,000 vectors.
   - Compare directly with `filtered-ann-research-medium` reference metrics (which used $N=100,000$).
5. **Checkpoint 5 (250K / 500K Decisions)**:
   - Rigorously compute projected memory for 250K and 500K using the measured empirical slope.
   - **Stop immediately** if projected peak RSS approaches available memory or if sustained swap thrashing occurs.
   - Transparently document the exact boundary where laptop hardware caps the experiment.

---

## 5. Experimental Protocols

- **Query Alignment**: Fixed official queries (IDs 0–24 for fast incremental passes) across EM, Range, and EMIS workloads.
- **Methods Evaluated**:
  1. `ExactSearch` (Exact filtered reference on the active subset prefix)
  2. `HNSW_PostFilter` ($M=16, ef_{\text{search}}=50$)
  3. `InGraphFilter` ($M=16, ef_{\text{search}}=50$)
  4. `PreFilter` (Inverted-index candidate filtering + BLAS distance)
  5. `AdaptiveRouter_2pct` (Evaluated on confirmed checkpoints)
- **Ground-Truth Rigor**: Ground truth is computed via ExactSearch on the evaluated subset prefix to ensure mathematical correctness of Recall@10.
- **Resource Logging**: Peak RSS recorded via `psutil.Process().memory_info().rss` at every stage into `resource_measurements.csv`.
