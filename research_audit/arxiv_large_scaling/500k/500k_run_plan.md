# 500K Scaling Experiment: Preflight Audit & Safety Gate Plan

**Project**: Filtered Approximate Nearest Neighbor Search (FANNS)  
**Dataset**: `SPCL/arxiv-for-fanns-large` ($D = 4,096$, float32 unit-normalized, 16,388 bytes/record)  
**Target Checkpoint**: $N = 500,000$ vectors  
**Host Architecture**: AMD Ryzen 3 5300U (4 Cores / 8 Threads @ 2.60 GHz), 8 GB Physical RAM (7.33 GB visible / 7,502 MB), Windows 11 Home, 8,192 MB Pagefile (`C:\pagefile.sys`), 273.76 GB NVMe SSD free space.  
**Execution Status**: **STOPPED — RESOURCE SAFETY LIMIT (INDEX CONSTRUCTION NOT STARTED)**

---

## 1. Phase 1 — Comprehensive Preflight Systems Inspection

### A. Windows Host Environment & Available Resources

| Resource | Measured Host Value | Notes & Analysis |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home (64-bit, Build 26100) | PowerShell shell, NT Kernel memory management |
| **CPU Model** | AMD Ryzen 3 5300U | 4 physical cores, 8 logical threads, base 2.60 GHz |
| **Total Physical RAM** | **7,502.1 MB (7.33 GB)** | Hardware reserved: ~666 MB from 8 GB physical module |
| **Currently Available RAM** | **1,791.9 MB (1.75 GB)** | Free memory available for new allocations |
| **Active Memory Usage** | **5,580.0 MB (5.58 GB / 76.1%)** | Consumed by OS, background services, IDE, browsers |
| **Windows Pagefile Total** | **8,192.0 MB (8.00 GB)** | Location: `C:\pagefile.sys` |
| **Windows Pagefile Used** | **564.3 MB (0.55 GB / 6.9%)** | Active swap pages currently committed |
| **Windows Pagefile Free** | **7,627.7 MB (7.45 GB)** | Remaining swap headroom |
| **System Commit Limit** | **15,694.1 MB (15.33 GB)** | Physical RAM + Pagefile total virtual address space |
| **Current Commit Usage** | **6,134.3 MB (6.13 GB / 39.1%)** | Total committed memory across all running processes |
| **Available Commit Margin** | **9,559.8 MB (9.20 GB)** | Headroom before total allocation failure (`STATUS_COMMITMENT_LIMIT`) |
| **Drive C: Free Space** | **273.76 GB** (Total: 475.74 GB) | Plentiful disk space for datasets and temporary artifacts |

### B. Active Memory-Intensive Background Processes

| Process Name | PID | Resident Memory (RSS) | Description |
| :--- | :--- | :--- | :--- |
| `MemCompression` | 3796 | 682.9 MB | Windows memory manager in-RAM compressed page store |
| `chrome.exe` | 1612 | 494.1 MB | Active web browser session |
| `brave.exe` | 6648 | 361.1 MB | Active web browser session |
| `Antigravity IDE.exe` | 17152 | 310.7 MB | Active research development environment |
| `chrome.exe` | 15792 | 240.8 MB | Browser helper / tab renderer |
| `MsMpEng.exe` | 5744 | 191.4 MB | Windows Defender Antivirus Service |
| `language_server` | 1532 | 184.1 MB | Python / IDE Language Server |
| `chrome.exe` | 14076 | 171.5 MB | Browser tab renderer |
| `PhoneExperienceHost`| 15788 | 135.3 MB | Windows Phone Link background service |
| `Antigravity IDE.exe` | 17976 | 131.1 MB | IDE language runtime |
| **Total Top 10 RSS** | — | **~2,903 MB (~2.84 GB)** | Baseline application footprint |

---

## 2. Dataset Specifications & Storage Requirements

### A. Record Sizing & Dimensionality
The dataset manifest from `SPCL/arxiv-for-fanns-large` defines:
- **Vector Dimension**: $D = 4,096$
- **Data Type**: `float32` (4 bytes per scalar)
- **File Format**: Standard `.fvecs` (4-byte integer dimension header per vector + $D \times 4$ float32 bytes)
- **Single Record Sizing**:
  $$\text{Bytes per vector} = 4 + (4,096 \times 4) = 16,388\text{ bytes} \approx 15.628\text{ KB/vector}$$

### B. 500K Checkpoint Raw Requirements
- **Total Vectors**: $N = 500,000$
- **Raw Vector Storage (`database_vectors.fvecs`)**:
  $$500,000 \times 16,388\text{ bytes} = 8,194,000,000\text{ bytes} = \mathbf{7.63\text{ GiB} \ (8,194.0\text{ MB})}$$
- **Metadata Storage (`database_attributes.jsonl`)**:
  500,000 JSON lines $\approx \mathbf{188.0\text{ MB}}$.
- **Total Source Data Disk Footprint**: $\mathbf{8.38\text{ GB}}$ (Safe: takes only 3.06% of available 273.76 GB disk space).

---

## 3. Implementation Code Inspection (`HNSWIndex`)

Inspection of `src/hnsw_index.py` from `filtered-ann-research-medium` reveals the following architectural behaviors:

1. **Vector Attachment**:
   - `set_vectors(vectors)` attaches `self.vectors = vectors`. When passed a memory-mapped array (`numpy.memmap`), it does **not** create a full in-memory copy upfront.
   - Precomputes `self.norms`: `np.zeros(500000, dtype=np.float32)` = **1.91 MB**. Computed in 5,000-vector chunks, creating temporary working slices of $5,000 \times 4,096 \times 4\text{ bytes} = \mathbf{81.92\text{ MB}}$.
2. **Graph Topology Allocation**:
   - Multi-layer adjacency lists are stored using native Python collections:
     `self.layers: List[Dict[int, List[int]]]`
   - Layer 0 contains **all 500,000 nodes**, where each node maps to a Python `list` of neighbor integer IDs (capped at $M_0 = 2 \times M = 32$).
   - Upper layers decay geometrically with factor $m_L = 1 / \ln(16) \approx 0.36$.
   - At $N=500,000$, the graph contains approximately $500,000 \times 32 \approx \mathbf{16,000,000\text{ directed edges}}$.
3. **Neighbor Pruning & Distance Recalculation**:
   - During node insertion (`_insert_point`), when neighbor count exceeds $M_{\text{max}}$, the implementation executes:
     ```python
     n_arr = np.array(n_list, dtype=np.int32)
     dots = np.dot(self.vectors[n_arr], self.vectors[neighbor])
     dists = self.norms[n_arr] + self.norms[neighbor] - 2.0 * dots
     part = np.argpartition(dists, m_max)[:m_max]
     sub_sort = np.argsort(dists[part])
     self.layers[lc][neighbor] = n_arr[part[sub_sort]].tolist()
     ```
   - Each neighbor update dynamically allocates temporary NumPy arrays and converts them back to Python lists via `.tolist()`.
4. **Memory-Mapping vs. Random Access Working Set**:
   - While `self.vectors` is memory-mapped, **HNSW beam search traverses the graph non-sequentially**.
   - As each new point is inserted, `_search_layer_base` accesses dozens of previously inserted nodes across layers to compute dot products.
   - Consequently, the operating system cache is forced to page in random 4KB memory-mapped vector chunks from disk across the entire 8.19 GB dataset.

---

## 4. Empirical Scaling Models Calibrated on 10K, 25K, 50K, 100K

### A. Measured Historical Checkpoints

| Checkpoint ($N$) | Raw Vectors | Serialized Index | Build Time | Peak RSS | RSS Delta | Free RAM Post-Build |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000** | 156.28 MB | 157.32 MB | 118.33 s (1.97 min) | 294.93 MB | 156.20 MB | 1,450.0 MB |
| **25,000** | 390.72 MB | 393.30 MB | 242.81 s (4.05 min) | 458.09 MB | 356.40 MB | 743.4 MB |
| **50,000** | 781.44 MB | 786.61 MB | 546.96 s (9.12 min) | 1,023.21 MB | 780.80 MB | 897.7 MB |
| **100,000** | 1,562.88 MB | 1,574.73 MB | 670.08 s (11.17 min) | 1,720.00 MB | 1,618.38 MB | **514.0 MB** |

### B. Mathematical Extrapolations for 500K

1. **Serialized Index Sizing**:
   $$\text{Slope} = \frac{1,574.73\text{ MB} - 157.32\text{ MB}}{90,000\text{ vectors}} = 0.015749\text{ MB/vector}$$
   $$\text{Projected Index File (500K)} = 500,000 \times 0.015749\text{ MB} = \mathbf{7,874.5\text{ MB} \ (7.69\text{ GB})}$$
2. **Process Resident Memory (RSS)**:
   - Growth from 50K to 100K: $\Delta \text{RSS} = 1,720.0 - 1,023.2 = 696.8\text{ MB}$ for 50K vectors ($13.936\text{ KB/vector}$).
   - Extrapolating 400K additional vectors beyond 100K:
     $$\text{Additional RSS} = 400,000 \times 13.936\text{ KB} = 5,574.4\text{ MB}$$
     $$\mathbf{\text{Projected Peak RSS (500K)}} = 1,720.0\text{ MB} + 5,574.4\text{ MB} = \mathbf{7,294.4\text{ MB} \ (7.12\text{ GB})}$$
3. **Index Construction Time**:
   - Calibrated $O(N \log N)$ complexity constant: $C \approx 4.02 \times 10^{-4}\text{ s/(vector}\cdot\text{bit)}$.
   $$\text{Clean Build Time (500K)} = C \times 500,000 \times \log_2(500,000) \approx 3,819\text{ seconds} \approx \mathbf{63.7\text{ minutes}}$$
   *(Note: This projection assumes zero disk page thrashing; active paging increases this to $>15–20$ hours).*

---

## 5. Mandatory Safety Gate & Abort Threshold Evaluation

### A. Predefined Abort Thresholds
1. **Physical RAM Safety Reserve**: Minimum **1,000 MB** of uncommitted physical RAM must remain available for operating system responsiveness.
2. **System Commit Limit**: System-wide committed memory must not exceed **85% of total commit limit** ($15,694 \times 0.85 = \mathbf{13,340\text{ MB}}$).
3. **Physical Memory Deficit**: The estimated process RSS must not exceed the total available physical RAM ($1,791.9\text{ MB}$).
4. **Paging Thrashing Risk**: Sustained hard page fault rates must not dominate execution cycles.

### B. Safety Gate Evaluation Matrix

| Metric / Check | Required Safe Value | Measured / Projected 500K Value | Evaluation Verdict |
| :--- | :--- | :--- | :--- |
| **Disk Space Available** | $> 30\text{ GB}$ reserve | **273.76 GB** free | **PASS** (Low risk) |
| **Physical RAM Deficit** | Peak RSS $\le 1,791.9\text{ MB}$ | **7,294.4 MB** ($4.07\times$ available RAM) | **FAIL — SEVERE DEFICIT** |
| **Commit Usage at Peak** | $\le 13,340\text{ MB}$ (85% limit) | **13,428.7 MB** ($85.6\%$ to $90.4\%$) | **FAIL — EXCEEDS SAFETY LIMIT** |
| **DRAM vs Working Set** | Working Set $\le 7,502\text{ MB}$ | **$\ge 12,500\text{ MB}$** (Graph + Vectors) | **FAIL — THRASHING GUARANTEED** |
| **Estimated Build Paging** | Minimal swap overhead | $> 5.5\text{ GB}$ forced to SSD pagefile | **FAIL — SYSTEM UNRESPONSIVE** |

### C. Formal Safety Gate Decision
$$\mathbf{SAFETY \ GATE \ STATUS: \ FAILED \ (CRITICAL \ RESOURCE \ RISK)}$$
$$\mathbf{ACTION: \ DO \ NOT \ START \ INDEX \ CONSTRUCTION}$$

**Technical Justification**:
Proceeding with index construction on 500,000 vectors with the current pure-Python graph engine on an 8GB laptop would force at least 5.5 GB of active process memory into `C:\pagefile.sys`. Because HNSW insertion relies on millions of non-sequential random hops across both the graph and the 8.19 GB vector memory map, execution would instantly collapse into continuous NVMe SSD thrashing ($1,000\times$ latency penalty). This violates the core task constraint: *"Abort before an estimated or observed memory exhaustion event. Stop if the system begins sustained heavy swapping, becomes unresponsive, or crosses a predefined memory limit."*

---

## 6. Minimum Engineering Changes Required for a Reasonable 500K Run

To make a 500K benchmark feasible and safe, the following modifications are strictly required:

1. **Native C++ Graph Backend (`hnswlib` or Faiss)**:
   - In pure Python, storing 16 million edges using `list` objects of `int` requires ~5.5–7.3 GB of memory.
   - In native C++ (`hnswlib`), edges are packed into contiguous flat arrays of 32-bit integers (`int32_t`), reducing graph topology overhead from **7.3 GB down to ~64 MB**.
2. **Vector Quantization (Scalar Quantization SQ8 / Product Quantization PQ)**:
   - Quantizing 4,096-dim float32 vectors to 8-bit integers compresses vector storage by 75% (from **8.19 GB down to 2.05 GB**), allowing the vectors to fit comfortably in RAM.
3. **Dedicated Compute Hardware**:
   - Minimum **32 GB to 64 GB ECC RAM** on a dedicated Linux compute node (e.g. AWS `c6i.4xlarge` or institutional cluster node).
