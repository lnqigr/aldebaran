"""
Standalone utility to inspect and safely purge Lightkurve and Astropy FITS caches.
Frees up disk space without touching results, logs, or reports.
"""

import os
import shutil
import sys


def get_dir_size_gb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(root, file))
        for root, _, files in os.walk(path)
        for file in files
    )
    return total / (1024.0 ** 3)


def clean_cache(dry_run: bool = False):
    cache_dirs = [
        os.path.join(os.path.expanduser("~"), ".cache", "lightkurve"),
        os.path.join(os.path.expanduser("~"), ".lightkurve-cache"),
        os.path.join(os.path.expanduser("~"), ".astropy", "cache", "download"),
    ]

    total_freed = 0.0
    for c_dir in cache_dirs:
        if os.path.exists(c_dir):
            sz = get_dir_size_gb(c_dir)
            print(f"Found cache: {c_dir} ({sz:.2f} GB)")
            if not dry_run:
                try:
                    shutil.rmtree(c_dir, ignore_errors=True)
                    os.makedirs(c_dir, exist_ok=True)
                    print(f"  [OK] Cleared {sz:.2f} GB from {c_dir}")
                    total_freed += sz
                except Exception as e:
                    print(f"  [!] Failed to clear {c_dir}: {e}")
            else:
                total_freed += sz

    if total_freed > 0:
        action = "Would free" if dry_run else "Successfully freed"
        print(f"\n{action} {total_freed:.2f} GB of cache.")
    else:
        print("[OK] All download caches are clean.")


if __name__ == "__main__":
    is_dry = "--dry-run" in sys.argv
    clean_cache(dry_run=is_dry)
