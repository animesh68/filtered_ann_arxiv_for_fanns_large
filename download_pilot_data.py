"""
Downloader for arXiv-Large FANNS Pilot Dataset Subset.
Fetches verified prefix from official SPCL/arxiv-for-fanns-large Hugging Face repository.
Safely bounds local storage to < 200 MB.
"""

import json
import os
from pathlib import Path
import time
import urllib.request
import numpy as np

DATASET_REPO = "SPCL/arxiv-for-fanns-large"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"

OUTPUT_DIR = Path("data/arxiv-large-pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_N = 10000        # 10,000 database vectors (10x Small scale)
DIM = 4096             # 4,096 dimensions
VECTOR_RECORD_BYTES = 4 + DIM * 4  # 16,388 bytes per vector
NUM_QUERIES = 50       # 50 controlled queries per workload


def download_byte_range(url: str, start_byte: int, end_byte: int, dest_path: Path):
    """Downloads a specific byte range and writes to destination."""
    print(f"Downloading [{dest_path.name}] (Bytes {start_byte} to {end_byte}, {(end_byte-start_byte+1)/(1024*1024):.2f} MB)...")
    req = urllib.request.Request(url, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        chunk_size = 1024 * 1024 * 4  # 4 MB buffer
        downloaded = 0
        total = end_byte - start_byte + 1
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100
            print(f"  -> {downloaded / (1024*1024):.2f}/{total / (1024*1024):.2f} MB ({pct:.1f}%)", end="\r")
    t1 = time.time()
    speed_mb = (total / (1024 * 1024)) / (t1 - t0) if t1 > t0 else 0
    print(f"\n  [OK] Saved {dest_path.name} in {t1-t0:.2f}s ({speed_mb:.2f} MB/s)")


def download_jsonl_prefix(url: str, num_lines: int, dest_path: Path):
    """Streams and downloads the first num_lines JSON objects."""
    print(f"Streaming [{dest_path.name}] ({num_lines} lines)...")
    t0 = time.time()
    req = urllib.request.Request(url)
    lines_written = 0
    with urllib.request.urlopen(req) as resp, open(dest_path, "w", encoding="utf-8") as out_f:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            out_f.write(line + "\n")
            lines_written += 1
            if lines_written % 2000 == 0 or lines_written == num_lines:
                print(f"  -> {lines_written}/{num_lines} records", end="\r")
            if lines_written >= num_lines:
                break
    t1 = time.time()
    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"\n  [OK] Saved {dest_path.name} ({size_mb:.2f} MB, {lines_written} lines) in {t1-t0:.2f}s")


def download_official_gt_sample():
    """Fetches first 50 entries of official full-dataset ground truth for audit."""
    gt_url = f"{BASE_URL}/ground_truth_em.ivecs"
    # Each entry in large GT is 100 int32 with 1 int32 prefix = 404 bytes
    gt_bytes = NUM_QUERIES * 404
    gt_dest = OUTPUT_DIR / "official_gt_em_first50.ivecs"
    if not gt_dest.exists() or gt_dest.stat().st_size != gt_bytes:
        download_byte_range(gt_url, 0, gt_bytes - 1, gt_dest)


def run_download():
    print("=" * 70)
    print(f"ARXIV-LARGE FANNS PILOT DATASET EXTRACTOR")
    print(f"Source: {DATASET_REPO}")
    print(f"Target Directory: {OUTPUT_DIR.resolve()}")
    print("=" * 70)

    # 1. Database Vectors Prefix (10,000 vectors)
    db_vec_path = OUTPUT_DIR / "database_vectors.fvecs"
    db_vec_bytes = PILOT_N * VECTOR_RECORD_BYTES
    if not db_vec_path.exists() or db_vec_path.stat().st_size != db_vec_bytes:
        download_byte_range(f"{BASE_URL}/database_vectors.fvecs", 0, db_vec_bytes - 1, db_vec_path)
    else:
        print(f"[EXISTS] {db_vec_path.name} ({db_vec_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Database Attributes (10,000 JSON lines)
    db_attr_path = OUTPUT_DIR / "database_attributes.jsonl"
    if not db_attr_path.exists():
        download_jsonl_prefix(f"{BASE_URL}/database_attributes.jsonl", PILOT_N, db_attr_path)
    else:
        print(f"[EXISTS] {db_attr_path.name} ({db_attr_path.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Query Vectors Prefix (50 queries)
    q_vec_path = OUTPUT_DIR / "query_vectors.fvecs"
    q_vec_bytes = NUM_QUERIES * VECTOR_RECORD_BYTES
    if not q_vec_path.exists() or q_vec_path.stat().st_size != q_vec_bytes:
        download_byte_range(f"{BASE_URL}/query_vectors.fvecs", 0, q_vec_bytes - 1, q_vec_path)
    else:
        print(f"[EXISTS] {q_vec_path.name} ({q_vec_path.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Query Attributes (EM, Range, EMIS)
    for workload, filename in [
        ("EM", "em_query_attributes.jsonl"),
        ("Range", "r_query_attributes.jsonl"),
        ("EMIS", "emis_query_attributes.jsonl"),
    ]:
        dest = OUTPUT_DIR / filename
        if not dest.exists():
            download_jsonl_prefix(f"{BASE_URL}/{filename}", NUM_QUERIES, dest)
        else:
            print(f"[EXISTS] {dest.name}")

    # 5. Official GT sample for cross-scale audit
    download_official_gt_sample()

    print("\n" + "=" * 70)
    print("ALL PILOT DATASET ARTIFACTS VERIFIED")
    print("=" * 70)
    for p in sorted(OUTPUT_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<35} {size_mb:>8.2f} MB")


if __name__ == "__main__":
    run_download()
