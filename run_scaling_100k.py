"""
arXiv-Large 100K Scaling Benchmark and Integrated Multi-Checkpoint Pipeline.
Executes 100K checkpoint, logs router dispatch decisions, consolidates 10K, 25K, 50K, and 100K records,
and writes updated audit artifacts.
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
UPDATED_LOG_FILE = AUDIT_DIR / "run_log_updated.txt"


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


def run_100k_checkpoint():
    sys.stdout = Logger(UPDATED_LOG_FILE)
    print("\n" + "=" * 80)
    print("STARTING 100,000-VECTOR SCALING CHECKPOINT EXECUTION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    TARGET_N = 100000
    NUM_QUERIES = 25

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
    assert all_vectors.shape[0] >= TARGET_N, f"File has {all_vectors.shape[0]} vectors, need {TARGET_N}"
    db_vectors = all_vectors[:TARGET_N]

    all_attrs = load_jsonl(attr_path)
    assert len(all_attrs) >= TARGET_N, f"Attributes has {len(all_attrs)} lines, need {TARGET_N}"
    db_attributes = all_attrs[:TARGET_N]

    query_vectors = read_fvecs(SCALING_DATA_DIR / "query_vectors.fvecs", mmap=False)[:NUM_QUERIES]
    em_attrs = load_jsonl(SCALING_DATA_DIR / "em_query_attributes.jsonl")[:NUM_QUERIES]
    r_attrs = load_jsonl(SCALING_DATA_DIR / "r_query_attributes.jsonl")[:NUM_QUERIES]
    emis_attrs = load_jsonl(SCALING_DATA_DIR / "emis_query_attributes.jsonl")[:NUM_QUERIES]

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
    print(f"[HNSW BUILD] Constructing graph on {TARGET_N:,} vectors (M={M}, efC={EF_CONSTRUCTION})...")
    hnsw.build_index(db_vectors, progress_bar=True)
    t1_build = time.time()
    build_time_s = t1_build - t0_build

    ram_post_build_rss_mb = proc.memory_info().rss / (1024 * 1024)
    peak_rss_mb = ram_post_build_rss_mb
    print(f"[HNSW COMPLETE] Build Time: {build_time_s:.2f}s ({build_time_s/60:.2f} min)")
    print(f"                Post-Build RSS: {ram_post_build_rss_mb:.1f} MB (Delta: {ram_post_build_rss_mb - ram_initial_rss_mb:.1f} MB)")
    print(f"                Graph Stats: Enter Point = {hnsw.enter_point}, Max Layer = {hnsw.max_layer}")

    # Serialize index
    idx_filename = AUDIT_DIR / f"hnsw_index_100k.pkl"
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
    for q_idx in range(min(5, NUM_QUERIES)):
        q_vec = query_vectors[q_idx]
        pred = parse_query_filter("EM", em_attrs[q_idx])
        gt_ids, _, _ = exact_searcher.search(q_vec, pred, k=10)
        ig_ids, _, _ = ingraph_searcher.search(q_vec, pred, k=10, ef_search=EF_SEARCH)
        valid_mask = meta_idx.get_boolean_mask(pred)
        assert all(valid_mask[cid] for cid in ig_ids), f"Predicate violation on query {q_idx}!"
        rec = compute_recall_at_k(ig_ids, gt_ids, k=10)
        print(f"  Q{q_idx}: Cand Count = {len(meta_idx.get_matching_ids(pred)):<6}, GT = {len(gt_ids)}, InGraph Recall@10 = {rec:.2f}")

    print("[SMOKE TEST PASSED] 100% Predicate compliance verified.")

    # 6. Controlled Benchmark Evaluation
    workloads = [
        ("EM", em_attrs, "EM"),
        ("Range", r_attrs, "Range"),
        ("EMIS", emis_attrs, "EMIS"),
    ]

    records_100k: List[Dict[str, Any]] = []
    t0_bench = time.time()
    print(f"[BENCHMARK] Evaluating {NUM_QUERIES} queries x 3 workloads x 5 methods...")

    for w_name, attrs, f_type in workloads:
        for q_id in range(NUM_QUERIES):
            q_vec = query_vectors[q_id]
            pred = parse_query_filter(f_type, attrs[q_id])
            matching_ids = meta_idx.get_matching_ids(pred)
            cand_count = len(matching_ids)
            selectivity = float(cand_count) / float(TARGET_N)
            valid_mask = meta_idx.get_boolean_mask(pred)

            # 1. ExactSearch (Ground Truth)
            with PrecisionTimer() as t_exact:
                gt_ids, gt_dists, _ = exact_searcher.search(q_vec, pred, k=10)
            records_100k.append({
                "dataset_size": TARGET_N,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "ExactSearch",
                "router_dispatch": "N/A",
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
            records_100k.append({
                "dataset_size": TARGET_N,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "HNSW_PostFilter",
                "router_dispatch": "N/A",
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
            records_100k.append({
                "dataset_size": TARGET_N,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "InGraphFilter",
                "router_dispatch": "N/A",
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
            records_100k.append({
                "dataset_size": TARGET_N,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "PreFilter",
                "router_dispatch": "N/A",
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
            records_100k.append({
                "dataset_size": TARGET_N,
                "workload": w_name,
                "query_id": q_id,
                "selectivity": selectivity,
                "candidate_count": cand_count,
                "method": "AdaptiveRouter_2pct",
                "router_dispatch": ar_decision,
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

    print(f"[BENCHMARK COMPLETE] {len(records_100k)} queries executed in {bench_time_s:.2f}s")
    print(f"[RESOURCE STATS] Peak RSS: {peak_rss_mb:.1f} MB, Free System RAM Remaining: {free_ram_end_mb:.1f} MB")

    # -------------------------------------------------------------
    # CONSOLIDATE ALL CHECKPOINTS (10K, 25K, 50K, 100K) INTO UPDATED CSVs
    # -------------------------------------------------------------
    print("\n[CONSOLIDATING] Creating unified multi-checkpoint datasets (10K, 25K, 50K, 100K)...")

    # 1. Pilot 10K (first 25 queries)
    pilot_raw_path = Path("research_audit/arxiv_large_pilot/pilot_raw_results.csv")
    df_pilot = pd.read_csv(pilot_raw_path)
    df_pilot_25q = df_pilot[df_pilot["query_id"] < NUM_QUERIES].copy()
    df_pilot_25q["dataset_size"] = 10000
    df_pilot_25q["router_dispatch"] = df_pilot_25q.apply(
        lambda r: ("PreFilter" if r["selectivity"] <= 0.02 else "InGraphFilter") if r["method"] == "AdaptiveRouter_2pct" else "N/A",
        axis=1
    )
    df_pilot_25q["visited_count"] = df_pilot_25q["visited_or_eval_count"]
    cols_order = [
        "dataset_size", "workload", "query_id", "selectivity", "candidate_count",
        "method", "router_dispatch", "k", "recall_at_10", "latency_ms",
        "results_returned", "valid_results", "visited_count", "error"
    ]
    df_pilot_clean = df_pilot_25q[cols_order]

    # 2. Previous 25K and 50K scaling results
    old_scaling_path = AUDIT_DIR / "scaling_raw_results.csv"
    df_old_scaling = pd.read_csv(old_scaling_path)
    df_old_scaling["router_dispatch"] = df_old_scaling.apply(
        lambda r: ("PreFilter" if r["selectivity"] <= 0.02 else "InGraphFilter") if r["method"] == "AdaptiveRouter_2pct" else "N/A",
        axis=1
    )
    df_old_clean = df_old_scaling[cols_order]

    # 3. New 100K results
    df_100k = pd.DataFrame(records_100k)[cols_order]

    # Combine all
    df_all_updated = pd.concat([df_pilot_clean, df_old_clean, df_100k], ignore_index=True)
    UPDATED_RAW_CSV = AUDIT_DIR / "scaling_raw_results_updated.csv"
    df_all_updated.to_csv(UPDATED_RAW_CSV, index=False)
    print(f"[SAVED] {UPDATED_RAW_CSV.name} ({len(df_all_updated)} total method-query records)")

    # Compute updated summary
    summary_rows = []
    for (sz, m_name, w_name), grp in df_all_updated.groupby(["dataset_size", "method", "workload"]):
        lats = grp["latency_ms"].values
        recs = grp["recall_at_10"].values
        sels = grp["selectivity"].values
        cands = grp["candidate_count"].values
        tot_time_s = np.sum(lats) / 1000.0
        qps = len(lats) / tot_time_s if tot_time_s > 0 else 0.0

        # Dispatch counts for router
        dispatches = grp["router_dispatch"].value_counts().to_dict() if m_name == "AdaptiveRouter_2pct" else {}

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
            "router_prefilter_dispatches": dispatches.get("PreFilter", 0),
            "router_ingraph_dispatches": dispatches.get("InGraphFilter", 0),
        })

    df_summary_updated = pd.DataFrame(summary_rows)
    UPDATED_SUMMARY_CSV = AUDIT_DIR / "scaling_summary_updated.csv"
    df_summary_updated.to_csv(UPDATED_SUMMARY_CSV, index=False)
    print(f"[SAVED] {UPDATED_SUMMARY_CSV.name}")

    # Update resource measurements
    res_100k = {
        "dataset_size": TARGET_N,
        "raw_vector_file_mb": (TARGET_N * 16388) / (1024 * 1024),
        "index_build_time_s": build_time_s,
        "index_file_size_mb": idx_size_mb,
        "peak_rss_mb": peak_rss_mb,
        "ram_delta_mb": peak_rss_mb - ram_initial_rss_mb,
        "free_ram_start_mb": free_ram_start_mb,
        "free_ram_end_mb": free_ram_end_mb,
        "query_benchmark_time_s": bench_time_s,
        "num_queries": len(records_100k),
    }

    # Consolidated resource table
    df_old_res = pd.read_csv(AUDIT_DIR / "resource_measurements.csv")
    res_10k = {
        "dataset_size": 10000,
        "raw_vector_file_mb": 163.88,
        "index_build_time_s": 118.33,
        "index_file_size_mb": 157.32,
        "peak_rss_mb": 294.93,
        "ram_delta_mb": 156.20,
        "free_ram_start_mb": 1510.0,
        "free_ram_end_mb": 1450.0,
        "query_benchmark_time_s": 24.69,
        "num_queries": 750,
    }
    df_all_res = pd.concat([pd.DataFrame([res_10k]), df_old_res, pd.DataFrame([res_100k])], ignore_index=True)
    df_all_res = df_all_res.drop_duplicates(subset=["dataset_size"], keep="last").sort_values("dataset_size")
    UPDATED_RES_CSV = AUDIT_DIR / "resource_measurements_updated.csv"
    df_all_res.to_csv(UPDATED_RES_CSV, index=False)
    print(f"[SAVED] {UPDATED_RES_CSV.name}")

    # Print summary table for 100K
    print("\n" + "=" * 95)
    print("SUMMARY FOR CHECKPOINT N = 100,000")
    print("=" * 95)
    for _, r in df_summary_updated[df_summary_updated["dataset_size"] == TARGET_N].iterrows():
        disp_info = f" [Pre:{r['router_prefilter_dispatches']}, IG:{r['router_ingraph_dispatches']}]" if "Router" in r["method"] else ""
        print(f"{r['method']:<20} {r['workload']:<8} Recall: {r['mean_recall']:<8.4f} Latency: {r['mean_latency_ms']:<8.2f} ms (P95: {r['p95_latency_ms']:<8.2f} ms) QPS: {r['qps']:<6.1f}{disp_info}")
    print("=" * 95)


if __name__ == "__main__":
    run_100k_checkpoint()
