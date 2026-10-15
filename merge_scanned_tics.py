#!/usr/bin/env python3
"""
ALDEBARAN UTILITY: MERGE CLUSTER & LOCAL SCANNED TICS
Aggregates scanned TIC IDs from MetaCentrum cluster runs, local archives,
and active workspace into a deduplicated, numerically sorted master catalog.
"""

import os
import glob
import sys

def merge_scanned_tics(remote_file=None):
    results_dir = "tess_results"
    local_master = os.path.join(results_dir, "scanned_tics.txt")
    
    all_tics = set()
    sources = []
    
    # 1. Existing local master
    if os.path.exists(local_master):
        with open(local_master, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            all_tics.update(lines)
            sources.append((local_master, len(lines)))
            
    # 2. Archive v75 history
    archive_v75 = os.path.join(results_dir, "archive", "archive_v75", "scanned_tics.txt")
    if os.path.exists(archive_v75):
        with open(archive_v75, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            all_tics.update(lines)
            sources.append((archive_v75, len(lines)))

    # 3. Specified remote file or any cluster_scanned_tics*.txt
    candidate_files = []
    if remote_file and os.path.exists(remote_file):
        candidate_files.append(remote_file)
    
    candidate_files.extend(glob.glob(os.path.join(results_dir, "*scanned*remote*.txt")))
    candidate_files.extend(glob.glob(os.path.join(results_dir, "*cluster*scanned*.txt")))
    candidate_files.extend(glob.glob(os.path.join(results_dir, "scanned_tics_cluster.txt")))
    candidate_files.extend(glob.glob(os.path.join(results_dir, "sector_*", "scanned_tics.txt")))

    for fpath in sorted(set(candidate_files)):
        if os.path.exists(fpath) and fpath != local_master:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
                all_tics.update(lines)
                sources.append((fpath, len(lines)))

    print("=" * 65)
    print(" ALDEBARAN SCANNED TICS CONSOLIDATION SUMMARY")
    print("=" * 65)
    for src, cnt in sources:
        print(f" [+] Source: {src:<50} ({cnt:>6} TICs)")
        
    # Sort numerically
    def sort_key(x):
        try:
            return int(x)
        except ValueError:
            return x

    sorted_tics = sorted(all_tics, key=sort_key)
    
    os.makedirs(results_dir, exist_ok=True)
    with open(local_master, "w", encoding="utf-8") as f:
        for tic in sorted_tics:
            f.write(f"{tic}\n")
            
    print("-" * 65)
    print(f" [*] Total unique scanned TICs in master catalog: {len(sorted_tics)}")
    print(f" [*] Successfully saved to: {local_master}")
    print("=" * 65)

if __name__ == '__main__':
    remote_arg = sys.argv[1] if len(sys.argv) > 1 else None
    merge_scanned_tics(remote_arg)
