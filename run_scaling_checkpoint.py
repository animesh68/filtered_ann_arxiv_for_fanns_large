"""
Scaling Benchmark Execution Engine for arXiv-Large FANNS.
Executes incremental checkpoints (25K, 50K, 100K, etc.) with real-time resource tracking,
systematic validation, and automated Stop/Go feasibility gating.
"""

import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# Add medium repository to path for establishing modules
MEDIUM_REPO_PATH = Path("C:/Users/anime/Desktop/filtered-ann-research-medium")
if str(MEDIUM_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(MEDIUM_REPO_PATH))

from src.dataset_loader import read_fvecs, load_jsonl
from src.filters import (
    MetadataIndex,
    FilterPredicate,
    EMFilterPredicate,
    RangeFilterPredicate,
    EMISFilterPredicate,
    parse_query_filter,
)
from src.exact_search import ExactFilteredSearcher
from src.hnsw_index import HNSWIndex
from src.postfilter import HNSWPostFilterSearcher
from src.ingraph import InGraphFilterSearcher
from src.prefilter import PreFilterSearcher
from src.metrics import compute_recall_at_k
from src.timing import PrecisionTimer

SCALING_DATA_DIR = Path("data/arxiv-large-scaling")
AUDIT_DIR = Path("research_audit/arxiv_large_scaling")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = AUDIT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = AUDIT_DIR / "run_log.txt"


class Logger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class AdaptiveRouterSearcher:
    """Selectivity-aware dynamic query router (threshold = 2%)."""
    def __init__(
        self,
        pre_searcher: PreFilterSearcher,
        ingraph_searcher: InGraphFilterSearcher,
        metadata_index: MetadataIndex,
        threshold: float = 0.02,
    ):
        self.pre_searcher = pre_searcher
        self.ingraph_searcher = ingraph_searcher
        self.metadata_index = metadata_index
        self.threshold = threshold

    def search(
        self,
        query_vec: np.ndarray,
        predicate: Optional[FilterPredicate],
        k: int = 10,
        ef_search: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, int, str]:
        matching_ids = self.metadata_index.get_matching_ids(predicate)
        total_docs = self.metadata_index.num_docs
        selectivity = float(len(matching_ids)) / float(total_docs) if total_docs > 0 else 0.0

        if selectivity <= self.threshold:
            decision = "PreFilter"
            topk_ids, topk_dists, evaluated_count = self.pre_searcher.search(
                query_vec=query_vec, predicate=predicate, k=k
            )
        else:
            decision = "InGraphFilter"
            topk_ids, topk_dists, evaluated_count = self.ingraph_searcher.search(
                query_vec=query_vec, predicate=predicate, k=k, ef_search=ef_search
            )
        return topk_ids, topk_dists, evaluated_count, decision


def run_scaling_checkpoint(target_n: int, num_queries_per_workload: int = 25) -> Dict[str, Any]:
    sys.stdout = Logger(LOG_FILE)
    print("\n" + "=" * 80)
    print(f"EXECUTING SCALING CHECKPOINT: N = {target_n:,} VECTORS")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    proc = psutil.Process()
    vm_before = psutil.virtual_memory()
    ram_initial_rss_mb = proc.memory_info().rss / (1024 * 1024)
    free_ram_start_mb = vm_before.available / (1024 * 1024)

    print(f"[RESOURCES PRE-RUN] Free System RAM: {free_ram_start_mb:.1f} MB, Process RSS: {ram_initial_rss_mb:.1f} MB")

    # 1. Load active prefix
    t0_load = time.time()
    vec_path = SCALING_DATA_DIR / "database_vectors.fvecs"
    attr_path = SCALING_DATA_DIR / "database_attributes.jsonl"

    all_vectors = read_fvecs(vec_path, mmap=True)
    assert all_vectors.shape[0] >= target_n, f"File has {all_vectors.shape[0]} vectors, need {target_n}"
    db_vectors = all_vectors[:target_n]

    all_attrs = load_jsonl(attr_path)
    assert len(all_attrs) >= target_n, f"Attributes has {len(all_attrs)} lines, need {target_n}"
    db_attributes = all_attrs[:target_n]

    query_vectors = read_fvecs(SCALING_DATA_DIR / "query_vectors.fvecs", mmap=False)[:num_queries_per_workload]
    em_attrs = load_jsonl(SCALING_DATA_DIR / "em_query_attributes.jsonl")[:num_queries_per_workload]
    r_attrs = load_jsonl(SCALING_DATA_DIR / "r_query_attributes.jsonl")[:num_queries_per_workload]
    emis_attrs = load_jsonl(SCALING_DATA_DIR / "emis_query_attributes.jsonl")[:num_queries_per_workload]

    print(f"[DATA LOADED] DB Vectors: {db_vectors.shape}, Attributes: {len(db_attributes)}, Queries: {query_vectors.shape[0]}")
    print(f"              Load elapsed: {time.time() - t0_load:.2f}s")

    # 2. Build Metadata Index
    t0_meta = time.time()
    meta_idx = MetadataIndex(db_attributes)
    t1_meta = time.time()
    print(f"[META INDEX] Constructed in {t1_meta - t0_meta:.3f}s for {meta_idx.num_docs} docs")

    # 3. Build HNSW Index
    M = 16
    EF_CONSTRUCTION = 50
    EF_SEARCH = 50
    RANDOM_SEED = 42

    hnsw = HNSWIndex(
        dimension=4096,
        metric="l2",
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=RANDOM_SEED,
    )

    t0_build = time.time()
    print(f"[HNSW BUILD] Constructing graph on {target_n:,} vectors (M={M}, efC={EF_CONSTRUCTION})...")
    hnsw.build_index(db_vectors, progress_bar=True)
    t1_build = time.time()
    build_time_s = t1_build - t0_build

    ram_post_build_rss_mb = proc.memory_info().rss / (1024 * 1024)
    peak_rss_mb = ram_post_build_rss_mb
    print(f"[HNSW COMPLETE] Build Time: {build_time_s:.2f}s ({build_time_s/60:.2f} min)")
    print(f"                Post-Build RSS: {ram_post_build_rss_mb:.1f} MB (Delta: {ram_post_build_rss_mb - ram_initial_rss_mb:.1f} MB)")
    print(f"                Graph Stats: Enter Point = {hnsw.enter_point}, Max Layer = {hnsw.max_layer}")

    # Serialize index
    idx_filename = AUDIT_DIR / f"hnsw_index_{target_n // 1000}k.pkl"
    with open(idx_filename, "wb") as f:
        pickle.dump(hnsw, f)
    idx_size_mb = idx_filename.stat().st_size / (1024 * 1024)
    print(f"[INDEX SAVED] {idx_filename.name} ({idx_size_mb:.2f} MB)")

    # 4. Initialize Searchers
    exact_searcher = ExactFilteredSearcher(db_vectors, meta_idx, metric="l2")
    post_searcher = HNSWPostFilterSearcher(hnsw, meta_idx)
    ingraph_searcher = InGraphFilterSearcher(hnsw, meta_idx)
    pre_searcher = PreFilterSearcher(db_vectors, meta_idx, metric="l2", precomputed_norms=hnsw.norms)
    router_2pct = AdaptiveRouterSearcher(pre_searcher, ingraph_searcher, meta_idx, threshold=0.02)

    # 5. Diagnostic 5-Query Smoke Test
    print("[SMOKE TEST] Running 5-query validation check...")
    for q_idx in range(min(5, num_queries_per_workload)):
        q_vec = query_vectors[q_idx]
        pred = parse_query_filter("EM", em_attrs[q_idx])
        gt_ids, _, _ = exact_searcher.search(q_vec, pred, k=10)
        ig_ids, _, _ = ingraph_searcher.search(q_vec, pred, k=10, ef_search=EF_SEARCH)
        valid_mask = meta_idx.get_boolean_mask(pred)
        assert all(valid_mask[cid] for cid in ig_ids), f"Predicate violation on query {q_idx}!"
        rec = compute_recall_at_k(ig_ids, gt_ids, k=10)
        print(f"  Q{q_idx}: Cand Count = {len(meta_idx.get_matching_ids(pred)):<5}, GT = {len(gt_ids)}, InGraph Recall@10 = {rec:.2f}")

    print("[SMOKE TEST PASSED] 100% Predicate compliance verified.")

    # 6. Controlled Benchmark Evaluation
    workloads = [
        ("EM", em_attrs, "EM"),
        ("Range", r_attrs, "Range"),
        ("EMIS", emis_attrs, "EMIS"),
    ]

    raw_records: List[Dict[str, Any]] = []
    t0_bench = time.time()
    print(f"[BENCHMARK] Evaluating {num_queries_per_workload} queries x 3 workloads x 5 methods...")

    for w_name, attrs, f_type in workloads:
        for q_id in range(num_queries_per_workload):
            q_vec = query_vectors[q_id]
            pred = parse_query_filter(f_type, attrs[q_id])
            matching_ids = meta_idx.get_matching_ids(pred)
            cand_count = len(matching_ids)
            selectivity = float(cand_count) / float(target_n)
            valid_mask = meta_idx.get_boolean_mask(pred)

            # 1. ExactSearch (Ground Truth)
            with PrecisionTimer() as t_exact:
                gt_ids, gt_dists, _ = exact_searcher.search(q_vec, pred, k=10)
            raw_records.append({
                "dataset_size": target_n,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "ExactSearch",
                "k": 10,
                "recall_at_10": 1.0,
                "latency_ms": t_exact.elapsed_ms,
                "results_returned": len(gt_ids),
                "valid_results": len(gt_ids),
                "visited_count": cand_count,
                "error": None,
            })

            # 2. HNSW_PostFilter
            with PrecisionTimer() as t_post:
                pf_ids, pf_dists, pf_eval = post_searcher.search(q_vec, pred, k=10, ef_search=EF_SEARCH)
            pf_valid = sum(1 for cid in pf_ids if valid_mask[cid])
            raw_records.append({
                "dataset_size": target_n,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "HNSW_PostFilter",
                "k": 10,
                "recall_at_10": compute_recall_at_k(pf_ids, gt_ids, k=10),
                "latency_ms": t_post.elapsed_ms,
                "results_returned": len(pf_ids),
                "valid_results": pf_valid,
                "visited_count": pf_eval,
                "error": None,
            })

            # 3. InGraphFilter
            with PrecisionTimer() as t_ig:
                ig_ids, ig_dists, ig_visited = ingraph_searcher.search(q_vec, pred, k=10, ef_search=EF_SEARCH)
            ig_valid = sum(1 for cid in ig_ids if valid_mask[cid])
            raw_records.append({
                "dataset_size": target_n,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "InGraphFilter",
                "k": 10,
                "recall_at_10": compute_recall_at_k(ig_ids, gt_ids, k=10),
                "latency_ms": t_ig.elapsed_ms,
                "results_returned": len(ig_ids),
                "valid_results": ig_valid,
                "visited_count": ig_visited,
                "error": None,
            })

            # 4. PreFilter
            with PrecisionTimer() as t_pre:
                pre_ids, pre_dists, pre_cands = pre_searcher.search(q_vec, pred, k=10)
            pre_valid = sum(1 for cid in pre_ids if valid_mask[cid])
            raw_records.append({
                "dataset_size": target_n,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "PreFilter",
                "k": 10,
                "recall_at_10": compute_recall_at_k(pre_ids, gt_ids, k=10),
                "latency_ms": t_pre.elapsed_ms,
                "results_returned": len(pre_ids),
                "valid_results": pre_valid,
                "visited_count": pre_cands,
                "error": None,
            })

            # 5. AdaptiveRouter (2%)
            with PrecisionTimer() as t_ar:
                ar_ids, ar_dists, ar_visited, ar_decision = router_2pct.search(q_vec, pred, k=10, ef_search=EF_SEARCH)
            ar_valid = sum(1 for cid in ar_ids if valid_mask[cid])
            raw_records.append({
                "dataset_size": target_n,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "AdaptiveRouter_2pct",
                "k": 10,
                "recall_at_10": compute_recall_at_k(ar_ids, gt_ids, k=10),
                "latency_ms": t_ar.elapsed_ms,
                "results_returned": len(ar_ids),
                "valid_results": ar_valid,
                "visited_count": ar_visited,
                "error": None,
            })

    t1_bench = time.time()
    bench_time_s = t1_bench - t0_bench
    vm_after = psutil.virtual_memory()
    free_ram_end_mb = vm_after.available / (1024 * 1024)
    peak_rss_mb = max(peak_rss_mb, proc.memory_info().rss / (1024 * 1024))

    print(f"[BENCHMARK COMPLETE] {len(raw_records)} queries executed in {bench_time_s:.2f}s")
    print(f"[RESOURCE STATS] Peak RSS: {peak_rss_mb:.1f} MB, Free System RAM Remaining: {free_ram_end_mb:.1f} MB")

    # 7. Update/Append CSV Artifacts
    df_new = pd.DataFrame(raw_records)
    RAW_CSV_PATH = AUDIT_DIR / "scaling_raw_results.csv"
    if RAW_CSV_PATH.exists():
        df_existing = pd.read_csv(RAW_CSV_PATH)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(RAW_CSV_PATH, index=False)

    # Compute Summary
    summary_rows = []
    for (sz, m_name, w_name), grp in df_all.groupby(["dataset_size", "method", "workload"]):
        lats = grp["latency_ms"].values
        recs = grp["recall_at_10"].values
        sels = grp["selectivity"].values
        cands = grp["candidate_count"].values
        tot_time_s = np.sum(lats) / 1000.0
        qps = len(lats) / tot_time_s if tot_time_s > 0 else 0.0

        summary_rows.append({
            "dataset_size": sz,
            "method": m_name,
            "workload": w_name,
            "num_queries": len(grp),
            "k": 10,
            "mean_recall": float(np.mean(recs)),
            "mean_latency_ms": float(np.mean(lats)),
            "median_latency_ms": float(np.median(lats)),
            "p95_latency_ms": float(np.percentile(lats, 95)),
            "p99_latency_ms": float(np.percentile(lats, 99)),
            "qps": float(qps),
            "mean_selectivity": float(np.mean(sels)),
            "mean_candidates": float(np.mean(cands)),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(AUDIT_DIR / "scaling_summary.csv", index=False)

    # Update Resource Measurements CSV
    res_entry = {
        "dataset_size": target_n,
        "raw_vector_file_mb": (target_n * 16388) / (1024 * 1024),
        "index_build_time_s": build_time_s,
        "index_file_size_mb": idx_size_mb,
        "peak_rss_mb": peak_rss_mb,
        "ram_delta_mb": peak_rss_mb - ram_initial_rss_mb,
        "free_ram_start_mb": free_ram_start_mb,
        "free_ram_end_mb": free_ram_end_mb,
        "query_benchmark_time_s": bench_time_s,
        "num_queries": len(raw_records),
    }
    RES_CSV_PATH = AUDIT_DIR / "resource_measurements.csv"
    if RES_CSV_PATH.exists():
        df_res = pd.read_csv(RES_CSV_PATH)
        df_res = pd.concat([df_res, pd.DataFrame([res_entry])], ignore_index=True)
    else:
        df_res = pd.DataFrame([res_entry])
    df_res.to_csv(RES_CSV_PATH, index=False)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"SUMMARY FOR CHECKPOINT N = {target_n:,}")
    print("=" * 90)
    for _, r in df_summary[df_summary["dataset_size"] == target_n].iterrows():
        print(f"{r['method']:<20} {r['workload']:<8} Recall: {r['mean_recall']:<8.4f} Latency: {r['mean_latency_ms']:<7.2f} ms (P95: {r['p95_latency_ms']:<7.2f} ms) QPS: {r['qps']:<6.1f}")
    print("=" * 90)

    checkpoint_stats = {
        "target_n": target_n,
        "build_time_s": build_time_s,
        "peak_rss_mb": peak_rss_mb,
        "free_ram_end_mb": free_ram_end_mb,
        "idx_size_mb": idx_size_mb,
        "bench_time_s": bench_time_s,
    }
    return checkpoint_stats


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    num_q = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    run_scaling_checkpoint(n, num_q)
