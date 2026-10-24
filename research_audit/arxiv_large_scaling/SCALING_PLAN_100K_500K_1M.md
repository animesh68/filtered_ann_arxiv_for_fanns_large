# Scaling Plan & Systems Feasibility Study: 100K / 500K / 1M Vectors

**Author**: Senior Systems Research Engineer  
**Date**: 2026-10-24  
**Project**: Filtered Approximate Nearest Neighbor Search (FANNS)  
**Dataset**: `SPCL/arxiv-for-fanns-large` (Iff et al., 2025)  
**Artifact Directory**: `research_audit/arxiv_large_scaling/`  

---

## 1. Phase 0 Audit & Reconciliation of Prior Findings

Prior to extending the benchmark, an exhaustive audit of `research_audit/arxiv_large_scaling/` and `research_audit/arxiv_large_pilot/` was conducted to reconcile measurements and algorithmic behavior.

### Issue A: Reconciliation of Measurement Counts
The previous scaling summary reported "750 queries executed", while the apparent experimental design ($3\text{ checkpoints} \times 3\text{ workloads} \times 25\text{ queries} \times 5\text{ methods}$) implied 1,125 method-query measurements.

**Audited Reconciliation**:
1. **Pilot Repository (`research_audit/arxiv_large_pilot/pilot_raw_results.csv`)**:
   - Evaluated exclusively at **$N = 10,000$**.
   - 50 unique queries $\times$ 3 workloads (EM, Range, EMIS) $\times$ 5 methods = **750 method-query rows**.
2. **Scaling Repository (`research_audit/arxiv_large_scaling/scaling_raw_results.csv`)**:
   - Evaluated at two checkpoints: **$N = 25,000$** and **$N = 50,000$**.
   - For each checkpoint: 25 unique queries (IDs 0–24) $\times$ 3 workloads $\times$ 5 methods = 375 rows.
   - Total rows in `scaling_raw_results.csv`: $375 + 375 = \mathbf{750\text{ method-query rows}}$.
3. **Clarification**:
   - The 10K pilot data was maintained in its isolated directory and was **not** merged into `scaling_raw_results.csv`.
   - All 750 method-query observations in `scaling_raw_results.csv` and all 750 in `pilot_raw_results.csv` were 100% successful with 0 failures, 0 timeouts, and 0 missing rows.
   - In this updated study, we maintain unified, append-only logs (`scaling_raw_results_updated.csv`) that combine 10K, 25K, 50K, and 100K with explicit checkpoint labeling.

### Issue B: Audit of Adaptive Router Dispatch Mechanics
The previous report noted that Range queries had a mean selectivity of $\approx 22.85\%$, yet `AdaptiveRouter_2pct` achieved a mean latency of $\approx 42\text{ ms}$ compared to InGraphFilter's $\approx 127\text{ ms}$.

**Code Inspection & Dispatch Audit**:
The router implementation in `run_scaling_checkpoint.py` evaluates query selectivity dynamically against a threshold $\tau = 0.02$ ($2\%$):
```python
matching_ids = self.metadata_index.get_matching_ids(predicate)
selectivity = float(len(matching_ids)) / float(total_docs)
if selectivity <= self.threshold:
    return self.pre_searcher.search(...)      # PreFilter
else:
    return self.ingraph_searcher.search(...)  # InGraphFilter
```

Auditing the per-query selectivity distribution across the 25 Range queries at $N=50,000$ revealed:
- **Mean Selectivity**: $25.73\%$ (driven by queries matching up to $91.8\%$ of documents).
- **Ultra-Sparse Outliers ($\sigma \le 2\%$)**: Exactly **4 out of 25 queries (16.0%)** have selectivity $\le 2\%$:
  - Query 0: $\sigma = 1.47\%$ (737 candidates)
  - Query 8: $\sigma = 1.73\%$ (863 candidates)
  - Query 10: $\sigma = 0.42\%$ (211 candidates)
  - Query 11: $\sigma = 2.00\%$ (1,000 candidates)

**The Cause of the Latency Discrepancy**:
- **InGraphFilter Pathological Wander**: On Query 10 ($\sigma = 0.42\%$), valid nodes are topologically sparse. InGraphFilter wanders through **37,793 nodes**, taking **1,277.95 ms**. On Query 0, 8, and 11, it takes $120 - 170\text{ ms}$. These 4 queries alone add over $1,738\text{ ms}$ to InGraphFilter's cumulative latency.
- **Router Offloading**: For these 4 queries, `AdaptiveRouter_2pct` selects `PreFilter`, evaluating Query 10 in **2.51 ms** (a **$509\times$ speedup**).
- For the remaining 21 dense queries ($\sigma > 2\%$), the router dispatches to `InGraphFilter`, which executes in $10 - 50\text{ ms}$ (where `PreFilter` would take $200 - 400\text{ ms}$).
- **Dispatch Proportions**:
  - **EM**: 0% PreFilter / 100% InGraphFilter (all queries have $\sigma \approx 32.3\%$)
  - **Range**: **16% PreFilter / 84% InGraphFilter** (4/25 sparse queries offloaded)
  - **EMIS**: **16% PreFilter / 84% InGraphFilter** (4/25 sparse queries offloaded)

This confirms that the router's lower mean latency is mathematically and algorithmically genuine: eliminating the long tail of graph wandering on sparse queries directly compresses workload mean latency.

### Issue C: Metric Fidelity & Correctness Validation
1. **Recall@10**: Computed against exact filtered ground truth (`ExactSearch`) evaluated on the identical active prefix.
2. **Predicate Compliance**: 100% compliance across all returned items (0 predicate violations).
3. **Timing Separation**: Index build time is strictly decoupled from per-query latencies.

---

## 2. Phase 1 Hardware Inspection & Feasibility Modeling

Verified directly from the operating system immediately prior to execution:

| System Parameter | Measured Value |
| :--- | :--- |
| **OS** | Microsoft Windows 11 Home (x64) |
| **CPU** | AMD Ryzen 3 5300U with Radeon Graphics (4 physical cores, 8 logical threads) |
| **Total Physical RAM** | 7,682,164 KB (**7.33 GB visible physical memory**) |
| **Available Free RAM** | 1,552,320 KB (**1.55 GB free physical memory**) |
| **Windows Pagefile** | `C:\pagefile.sys` (Allocated: **8,192 MB**, Current: 719 MB, Peak: 1,216 MB) |
| **Storage (Drive C:)** | 510.82 GB total, **299.89 GB free** |
| **Python Environment** | Python 3.14.6 (`numpy 2.5.3`, `pandas 3.0.6`, `scipy 1.18.1`, `psutil 7.2.2`) |
| **I/O Strategy** | `numpy.memmap` vector reading (mode='r') + streaming JSONL loader |

### Exact Vector Storage Formula
Each vector record in `.fvecs` contains 1 int32 dimension prefix ($4\text{ bytes}$) followed by $D = 4,096$ float32 components ($16,384\text{ bytes}$), totaling **16,388 bytes per vector**.
$$\text{Raw File Bytes} = N \times (4 + 4,096 \times 4) = N \times 16,388\text{ bytes}$$

---

## 3. Comprehensive Target Sizing & Feasibility Models

| Target Size ($N$) | Raw Disk File | In-Memory Graph RAM | Metadata & Index | Peak Build RSS | Projected Build Time | Feasibility Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **10K** | 163.88 MB | ~12 MB | ~4 MB | **295 MB** | 1.97 min | **Complete (Pilot Baseline)** |
| **25K** | 390.72 MB | ~30 MB | ~10 MB | **458 MB** | 4.05 min | **Complete (Scaling Checkpoint 1)** |
| **50K** | 781.44 MB | ~60 MB | ~20 MB | **1,023 MB** | 9.12 min | **Complete (Scaling Checkpoint 2)** |
| **100K** | 1,638.80 MB | ~120 MB | ~45 MB | **~1,760 MB** | **~12–15 min** | **MANDATORY CHECKPOINT: FEASIBLE** (Fits with 8GB pagefile backing) |
| **500K** | 8,194.00 MB | ~600 MB | ~220 MB | **~8,500–10,000 MB** | **$>5–10$ hours** | **HIGH RISK / RESOURCE BLOCKED** (Raw file alone exceeds laptop RAM) |
| **1M** | 16,388.00 MB | ~1,200 MB | ~450 MB | **$>18,000$ MB** | **$>15–24$ hours** | **DO NOT RUN LOCALLY** (Requires 64GB+ HPC Node) |

---

## 4. Resource Safety Policy & Execution Roadmap

1. **Phase 2 (100K Execution)**:
   - Extend dataset prefix from 50,000 to 100,000 vectors via HTTP byte-range request ($+781.44\text{ MB}$).
   - Stream additional 50,000 metadata records ($+16.86\text{ MB}$).
   - Build HNSW index ($M=16, ef_{\text{construction}}=50$, $100,000$ nodes).
   - Evaluate 25 queries $\times$ 3 workloads $\times$ 5 methods (**375 method-query evaluations**).
   - Record exact router dispatch decisions, recall, latency, and memory stats.
2. **Phase 3 (500K Gating Decision)**:
   - Evaluate measured build time and peak RSS at 100K.
   - If peak RSS crosses safety boundaries or Windows undergoes sustained paging, gate 500K and provide audited justification.
3. **Phase 4 (1M Assessment)**:
   - Formulate formal feasibility recommendation based on empirical scaling curves.
