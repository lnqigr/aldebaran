#!/usr/bin/env python3
"""
ALDEBARAN v8.0: Multi-Node Array Shard Reducer & Merger
Aggregates candidates, logs, and scanned TICs across all parallel sector shards,
deduplicates them, and triggers Pleiades Tier-2 deep vetting on the unified catalog.
"""
import glob
import os
import subprocess
import sys
import pandas as pd

def merge_shards(results_dir: str = "tess_results") -> None:
    print("=" * 70)
    print("ALDEBARAN v8.0: Aggregating Multi-Sector Array Shards")
    print("=" * 70)

    # 1. Merge Candidates Reports
    report_patterns = [
        os.path.join(results_dir, "sector_*", "candidates_report.csv"),
        os.path.join(results_dir, "candidates_report.csv")
    ]
    csv_files = []
    for pat in report_patterns:
        for f in glob.glob(pat):
            if os.path.exists(f) and os.path.getsize(f) > 50 and f not in csv_files:
                csv_files.append(f)

    if not csv_files:
        print("[!] No candidate CSV reports found across shards.")
    else:
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"[!] Warning reading {f}: {e}")

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            # Deduplicate by TIC ID, prioritizing the detection with highest SDE
            merged["sde_num"] = pd.to_numeric(merged["sde"], errors="coerce").fillna(0.0)
            merged = merged.sort_values(by="sde_num", ascending=False).drop_duplicates(subset=["tic_id"], keep="first")
            merged = merged.drop(columns=["sde_num"])

            out_csv = os.path.join(results_dir, "candidates_report.csv")
            merged.to_csv(out_csv, index=False)
            print(f"[+] Merged {len(merged)} unique candidates from {len(csv_files)} files -> {out_csv}")

    # 2. Merge Scanned TICs history
    all_scanned = set()
    root_history = os.path.join(results_dir, "scanned_tics.txt")
    if os.path.exists(root_history):
        with open(root_history, "r", encoding="utf-8") as f:
            all_scanned.update(line.strip() for line in f if line.strip())

    shard_histories = glob.glob(os.path.join(results_dir, "sector_*", "scanned_tics.txt"))
    for hf in shard_histories:
        with open(hf, "r", encoding="utf-8") as f:
            all_scanned.update(line.strip() for line in f if line.strip())

    with open(root_history, "w", encoding="utf-8") as f:
        for tic in sorted(all_scanned):
            f.write(f"{tic}\n")
    print(f"[+] Total unique scanned targets consolidated: {len(all_scanned)} -> {root_history}")

    # 3. Launch Pleiades Tier-2 Deep Vetting if candidates exist
    out_csv = os.path.join(results_dir, "candidates_report.csv")
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 50:
        print("\n[+] Launching Pleiades Tier-2 Deep Vetting on unified candidate list...")
        subprocess.run([sys.executable, "pleiades.py"], check=False)

        # 4. Launch Tier-3 CTOI Packager & Follow-Up Prioritizer
        if os.path.exists("ctoi_packager.py"):
            print("\n[+] Launching Tier-3 CTOI Packager & Follow-Up Prioritizer...")
            subprocess.run([sys.executable, "ctoi_packager.py"], check=False)
    else:
        print("\n[+] No candidates to vet. Tier-2 deep vetting skipped.")

    print("=" * 70)
    print("Shard aggregation and CTOI packaging complete!")
    print("=" * 70)

if __name__ == "__main__":
    merge_shards()
