"""
Incremental Dataset Extractor for arXiv-Large Scaling Experiments.
Extends the verified 10K dataset to 25K, 50K, and 100K checkpoints using HTTP range requests.
Reuses existing local prefix to minimize bandwidth and preserve official ordering.
"""

import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import numpy as np

DATASET_REPO = "SPCL/arxiv-for-fanns-large"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"

PILOT_DATA_DIR = Path("data/arxiv-large-pilot")
SCALING_DATA_DIR = Path("data/arxiv-large-scaling")
SCALING_DATA_DIR.mkdir(parents=True, exist_ok=True)

DIM = 4096
VECTOR_RECORD_BYTES = 4 + DIM * 4  # 16,388 bytes per vector


def download_byte_range(url: str, start_byte: int, end_byte: int, dest_path: Path, mode: str = "wb"):
    """Downloads a specific byte range and writes or appends to destination."""
    total_bytes = end_byte - start_byte + 1
    print(f"Downloading [{dest_path.name}] (Bytes {start_byte} to {end_byte}, {total_bytes/(1024*1024):.2f} MB)...")
    req = urllib.request.Request(url, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(dest_path, mode) as out_f:
        chunk_size = 1024 * 1024 * 4  # 4 MB buffer
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total_bytes * 100
            print(f"  -> {downloaded / (1024*1024):.2f}/{total_bytes / (1024*1024):.2f} MB ({pct:.1f}%)", end="\r")
    t1 = time.time()
    speed_mb = (total_bytes / (1024 * 1024)) / (t1 - t0) if t1 > t0 else 0
    print(f"\n  [OK] Downloaded {total_bytes / (1024*1024):.2f} MB in {t1-t0:.2f}s ({speed_mb:.2f} MB/s)")


def stream_jsonl_range(url: str, start_line: int, target_total_lines: int, dest_path: Path, mode: str = "w"):
    """Streams JSON lines from Hugging Face and appends up to target_total_lines."""
    lines_needed = target_total_lines - start_line
    print(f"Streaming [{dest_path.name}] (Lines {start_line} to {target_total_lines-1}, {lines_needed} lines needed)...")
    t0 = time.time()
    req = urllib.request.Request(url)
    current_line = 0
    lines_written = 0
    with urllib.request.urlopen(req) as resp, open(dest_path, mode, encoding="utf-8") as out_f:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if current_line >= start_line:
                out_f.write(line + "\n")
                lines_written += 1
                if lines_written % 5000 == 0 or current_line == target_total_lines - 1:
                    print(f"  -> {current_line + 1}/{target_total_lines} records", end="\r")
            current_line += 1
            if current_line >= target_total_lines:
                break
    t1 = time.time()
    print(f"\n  [OK] Extracted {lines_written} lines in {t1-t0:.2f}s")


def ensure_scaling_dataset(target_n: int):
    """Ensures scaling dataset has exact target_n vectors and metadata records."""
    print("=" * 70)
    print(f"ENSURING SCALING DATASET PREFIX: N = {target_n}")
    print("=" * 70)

    target_bytes = target_n * VECTOR_RECORD_BYTES
    vec_path = SCALING_DATA_DIR / "database_vectors.fvecs"
    attr_path = SCALING_DATA_DIR / "database_attributes.jsonl"

    # Copy query vectors and query attributes if not present
    for fname in ["query_vectors.fvecs", "em_query_attributes.jsonl", "r_query_attributes.jsonl", "emis_query_attributes.jsonl"]:
        dest = SCALING_DATA_DIR / fname
        src = PILOT_DATA_DIR / fname
        if not dest.exists() and src.exists():
            shutil.copy2(src, dest)
            print(f"[COPIED] {fname} from pilot")

    # 1. Vectors
    current_bytes = vec_path.stat().st_size if vec_path.exists() else 0
    if current_bytes < target_bytes:
        if current_bytes == 0 and (PILOT_DATA_DIR / "database_vectors.fvecs").exists():
            # Seed with 10K pilot
            pilot_bytes = (PILOT_DATA_DIR / "database_vectors.fvecs").stat().st_size
            shutil.copy2(PILOT_DATA_DIR / "database_vectors.fvecs", vec_path)
            current_bytes = pilot_bytes
            print(f"[SEEDED] database_vectors.fvecs seeded with 10K pilot ({pilot_bytes / (1024*1024):.2f} MB)")

        if current_bytes < target_bytes:
            start_b = current_bytes
            end_b = target_bytes - 1
            download_byte_range(
                url=f"{BASE_URL}/database_vectors.fvecs",
                start_byte=start_b,
                end_byte=end_b,
                dest_path=vec_path,
                mode="ab"
            )
    else:
        print(f"[EXISTS] database_vectors.fvecs already has {current_bytes / (1024*1024):.2f} MB (>= {target_bytes / (1024*1024):.2f} MB)")

    # 2. Metadata attributes
    current_lines = 0
    if attr_path.exists():
        with open(attr_path, "r", encoding="utf-8") as f:
            current_lines = sum(1 for _ in f)

    if current_lines < target_n:
        if current_lines == 0 and (PILOT_DATA_DIR / "database_attributes.jsonl").exists():
            shutil.copy2(PILOT_DATA_DIR / "database_attributes.jsonl", attr_path)
            with open(attr_path, "r", encoding="utf-8") as f:
                current_lines = sum(1 for _ in f)
            print(f"[SEEDED] database_attributes.jsonl seeded with 10K pilot ({current_lines} lines)")

        if current_lines < target_n:
            stream_jsonl_range(
                url=f"{BASE_URL}/database_attributes.jsonl",
                start_line=current_lines,
                target_total_lines=target_n,
                dest_path=attr_path,
                mode="a"
            )
    else:
        print(f"[EXISTS] database_attributes.jsonl already has {current_lines} lines (>= {target_n})")

    # Final verification
    final_bytes = vec_path.stat().st_size
    assert final_bytes >= target_bytes, f"Vector file too small: {final_bytes} < {target_bytes}"
    with open(attr_path, "r", encoding="utf-8") as f:
        final_lines = sum(1 for _ in f)
    assert final_lines >= target_n, f"Metadata lines too small: {final_lines} < {target_n}"

    print(f"[VERIFIED] Scaling prefix for N = {target_n} is ready on disk:")
    print(f"  Vectors:  {vec_path.name} ({final_bytes / (1024*1024):.2f} MB, {final_bytes // VECTOR_RECORD_BYTES} vectors)")
    print(f"  Metadata: {attr_path.name} ({attr_path.stat().st_size / (1024*1024):.2f} MB, {final_lines} records)")


if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    ensure_scaling_dataset(target)
