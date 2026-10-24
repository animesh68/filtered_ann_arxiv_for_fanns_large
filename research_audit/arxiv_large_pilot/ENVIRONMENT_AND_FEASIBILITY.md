# Environment and Feasibility Note: arXiv-Large FANNS Pilot

**Date**: 2026-10-24  
**Project**: Filtered Approximate Nearest Neighbor Search (FANNS)  
**Target Dataset**: `SPCL/arxiv-for-fanns-large` (Iff et al., 2025)  
**Execution Context**: Local Laptop (Research Presentation Pilot)

---

## 1. Actual Hardware Specifications

| Component | Verified Specification |
| :--- | :--- |
| **System / OS** | Microsoft Windows 11 Home (x64) |
| **CPU Model** | AMD Ryzen 3 5300U with Radeon Graphics |
| **Cores / Threads** | 4 Physical Cores / 8 Logical Processors |
| **Total Physical RAM** | 7,682,164 KB (~7.33 GB visible physical RAM / 8.0 GB installed) |
| **Available Free RAM** | ~1,583,444 KB (~1.51 GB free at start of inspection) |
| **Storage (Drive C:)** | Local Fixed Disk: 510.82 GB total, **302.85 GB free** |
| **Python Environment** | Python 3.14.6 (`C:\Users\anime\Desktop\filtered-ann-research-small\venv`) |
| **Core Libraries** | `numpy==2.5.3`, `pandas==3.0.6`, `scipy==1.18.1`, `matplotlib==3.11.2`, `psutil==7.2.2`, `huggingface_hub==2.1.1`, `tqdm==4.70.1` |
| **ANN Library** | Custom Python/NumPy modular HNSW research implementation (`HNSWIndex`), reused from `filtered-ann-research-medium/src` |

---

## 2. Official arXiv-Large Dataset Specifications

Verified from official repository manifest `SPCL/arxiv-for-fanns-large` on Hugging Face Hub:

| File Name | Size (Bytes) | Formatted Size | Content & Format |
| :--- | :--- | :--- | :--- |
| `database_vectors.fvecs` | 44,825,506,432 | **44.83 GB** | $N = 2,735,264$ vectors, $D = 4,096$, float32 unit-normalized |
| `database_attributes.jsonl` | 1,124,482,479 | **1.12 GB** | 2,735,264 paper metadata records (11 fields) |
| `query_vectors.fvecs` | 163,880,000 | **163.88 MB** | 10,000 queries, $D = 4,096$, float32 |
| `em_query_attributes.jsonl` | 130,000 | 130 KB | 10,000 Exact Match query attributes (`label`: integer) |
| `r_query_attributes.jsonl` | 430,000 | 430 KB | 10,000 Range query attributes (`range_start`, `range_end`) |
| `emis_query_attributes.jsonl` | 192,400 | 192 KB | 10,000 EMIS query attributes (`label`: string category) |
| `ground_truth.ivecs` | 4,040,000 | 4.04 MB | 10,000 queries $\times 100$ int32 unfiltered ground-truth IDs |
| `ground_truth_em.ivecs` | 4,040,000 | 4.04 MB | 10,000 queries $\times 100$ int32 EM filtered ground-truth IDs |
| `ground_truth_r.ivecs` | 4,040,000 | 4.04 MB | 10,000 queries $\times 100$ int32 Range filtered ground-truth IDs |
| `ground_truth_emis.ivecs` | 4,040,000 | 4.04 MB | 10,000 queries $\times 100$ int32 EMIS filtered ground-truth IDs |

### Raw Vector Storage Calculation
$$\text{Raw vector bytes} = N \times D \times 4 = 2,735,264 \times 4,096 \times 4 = 44,813,074,432\text{ bytes} \approx 44.81\text{ GB}$$
Each vector record has an additional 4-byte dimension prefix:
$$\text{Total file size} = 2,735,264 \times (4 + 4,096 \times 4) = 44,825,506,432\text{ bytes} = 44.825\text{ GB}$$

---

## 3. Memory Risks & Feasibility Assessment

1. **In-Memory Allocation Risk**:
   Loading the full 44.83 GB vector matrix into RAM is impossible on a system with 7.33 GB visible physical RAM. An in-memory load would immediately trigger an `OutOfMemoryError` or unrecoverable system freeze.
2. **Memory-Mapping (`np.memmap`) Limitation**:
   While `np.memmap` avoids immediate allocation, random memory access patterns during HNSW graph construction ($M=16, ef_{\text{construction}}=50$) and nearest neighbor searches across 44.83 GB will trigger continuous virtual memory page faults. On 7.33 GB RAM, the Windows paging system would thrash, causing the laptop to become unresponsive.
3. **Index Build Complexity**:
   Medium scale ($N=100,000$) required **713.9 seconds (11.9 minutes)** to build an HNSW index. Extrapolating $O(N \log N)$ to $N=2,735,264$ vectors ($27.35\times$ scale) yields an estimated construction time of $>25,000$ seconds (**~7.0 hours**), completely exceeding laptop capabilities and presentation preparation budgets.
4. **HPC Requirement**:
   As documented in the repository's supercomputer checklist (`hpc/hpc_checklist.md`), evaluation of the complete 2.735M dataset requires an HPC compute node with at least 64–128 GB RAM and 16–32 dedicated cores.

---

## 4. Go / No-Go Decision

- **Full-Dataset Local Evaluation ($N=2,735,264$)**: **NO-GO**. Unsafe on 8 GB laptop.
- **Resource-Aware Representative Pilot ($N_{\text{pilot}}=10,000$)**: **GO**. Scientifically sound, fully reproducible, and rigorously bounds RAM to $<500$ MB and index build time to $<1.5$ minutes.

---

## 5. Chosen Pilot Scope & Protocol

1. **Dataset Source**:
   Directly extract $N_{\text{pilot}} = 10,000$ database vectors and matching metadata records from the official `SPCL/arxiv-for-fanns-large` Hugging Face repository via HTTP byte-range requests.
2. **Dataset Scale**:
   - $N_{\text{pilot}} = 10,000$ vectors ($10\times$ the Small dataset of 1,000 vectors)
   - Dimensionality: $D = 4,096$
   - Raw vector storage: $10,000 \times 16,388\text{ bytes} = 163.88\text{ MB}$
   - Metadata storage: 10,000 JSON lines $\approx 4.1\text{ MB}$
3. **Query Workload**:
   - Official query vectors and attributes from `SPCL/arxiv-for-fanns-large`.
   - Query sample: 50 fixed queries (IDs 0–49) evaluated across all three official predicate regimes:
     - Exact Match (EM): `number_of_sub_categories == label`
     - Range: `range_start <= update_date <= range_end`
     - Exact Match In Set (EMIS): `label in main_categories`
4. **Ground-Truth Convention**:
   - Because the official `ground_truth_*.ivecs` IDs refer to vectors across the full 2.735M dataset, ground truth on the pilot subset is computed via **Exact Filtered Search (chunked BLAS)** directly on the pilot subset.
   - Verified Recall@10 is computed against this mathematical ground truth.
5. **Evaluated Methods**:
   - **Exact Filtered Search**: Mathematical ground-truth reference.
   - **HNSW PostFilter**: Naive post-filtering baseline ($M=16, ef_{\text{construction}}=50, ef_{\text{search}}=50$).
   - **InGraphFilter**: Single-graph predicate-aware traversal.
   - **PreFilter**: Inverted-index pruning followed by restricted vector scan.
   - **AdaptiveRouter**: Dynamic routing rule (if selectivity $\le 2\% \to \text{PreFilter}$, else $\text{InGraphFilter}$).
6. **Resource Budget**:
   - Maximum Peak RAM: $\le 600\text{ MB}$ (conservatively $<40\%$ of available free RAM)
   - Maximum Disk Usage: $\le 300\text{ MB}$
   - Index Build Time: $\le 90\text{ seconds}$
   - Query Run Time: $\le 120\text{ seconds}$
