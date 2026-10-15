#!/usr/bin/env python3
"""
search_toi_companions.py (Aldebaran v9.0)
========================================
Dedicated High-Yield Sibling Exoplanet Search Engine on Known NASA TOI Hosts.

Features:
- Ingests official TOI ephemerides (P, T0, T_dur, Depth) from local cache or NASA TAP.
- Cross-matches TOI hosts against SPOC 2-minute cadence light curves across specified sectors.
- In-transit masking ([-1.0*T_dur, +1.0*T_dur]) with linear flux interpolation prior to
  piecewise Savitzky-Golay detrending to prevent Gibbs edge ringing.
- Residual Box Least Squares (BLS) periodic search for shallow sub-Earth / super-Earth siblings.
- Protects against harmonic/alias false positives (m/n in {1, 1/2, 2, 1/3, 3, 2/3, 3/2, 1/4, 4}).
- Saves publication-grade discovery plots and candidate summaries.
"""

import os
import sys
import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aldebaran as ald

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TOI_Companions")


def parse_args():
    parser = argparse.ArgumentParser(description="Aldebaran TOI Sibling Exoplanet Search Engine")
    parser.add_argument("--sectors", type=int, nargs="+", default=[106, 105, 104, 103, 102, 101, 100, 99],
                        help="TESS sectors to search for TOI host companion planets")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--limit-per-sector", type=int, default=500, help="Max TOI hosts to screen per sector")
    parser.add_argument("--output-dir", type=str, default="tess_results/toi_companions",
                        help="Directory to save companion candidate reports and plots")
    return parser.parse_args()


def get_toi_hosts_in_sector(sector: int, toi_hosts_set: set) -> list:
    """Finds all TOI hosts with SPOC 2-minute light curves in the specified sector."""
    try:
        from astroquery.mast import Observations
        obs = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            dataproduct_type="timeseries"
        )
        if len(obs) == 0:
            return []
        df_obs = obs.to_pandas()
        spoc_mask = df_obs["target_name"].astype(str).str.isdigit() & (df_obs["provenance_name"] == "SPOC")
        spoc_tics = set(df_obs.loc[spoc_mask, "target_name"].astype(str).str.lstrip("0"))
        cleaned_host_set = {str(t).lstrip("0") for t in toi_hosts_set}
        matched = sorted(list(spoc_tics & cleaned_host_set))
        return matched
    except Exception as e:
        logger.warning("Error querying MAST for sector %d: %s", sector, e)
        return []


def process_toi_host(tic_id: str, sector: int) -> dict:
    """Executes masking and residual BLS search on a single TOI host star."""
    try:
        tic_out, disp, msg, cand_data = ald.analyze_single_target(tic_id, sector)
        if disp == "CANDIDATE" and cand_data is not None:
            logger.info(">>> SIBLING CANDIDATE DISCOVERED! TIC %s (Sector %d, SDE=%.2f, P=%.4f d)",
                        tic_id, sector, cand_data.get("sde", 0.0), cand_data.get("period_days", 0.0))
            return cand_data
        return None
    except Exception as e:
        logger.debug("Error processing TIC %s: %s", tic_id, e)
        return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ald.OUTPUT_DIR = args.output_dir
    report_file = os.path.join(args.output_dir, "toi_companions_summary.csv")

    logger.info("======================================================================")
    logger.info("ALDEBARAN v9.0 - TOI SIBLING EXOPLANET SEARCH ENGINE")
    logger.info("Target Sectors: %s | Workers: %d", args.sectors, args.workers)
    logger.info("Output Directory: %s", args.output_dir)
    logger.info("======================================================================")

    # 1. Load TOI catalog ephemerides
    known_tics, eb_tics, toi_ephem_dict = ald.load_known_catalogs_and_toi_ephemerides()
    toi_host_set = set(toi_ephem_dict.keys())
    logger.info("Loaded %d TOI host systems with cached orbital ephemerides.", len(toi_host_set))

    total_candidates = []

    for sec in args.sectors:
        logger.info("----------------------------------------------------------------------")
        logger.info("Scanning Sector %d for known TOI planetary hosts...", sec)
        toi_candidates = get_toi_hosts_in_sector(sec, toi_host_set)
        logger.info("Sector %d: Found %d TOI host systems with SPOC light curves.", sec, len(toi_candidates))

        if not toi_candidates:
            continue

        selected_tics = toi_candidates[:args.limit_per_sector]
        logger.info("Queueing %d TOI hosts for companion search across %d workers...", len(selected_tics), args.workers)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_toi_host, tic, sec): tic
                for tic in selected_tics
            }
            for fut in as_completed(futures):
                tic = futures[fut]
                try:
                    cand = fut.result()
                    if cand:
                        total_candidates.append(cand)
                        # Append immediately to report CSV
                        df_cand = pd.DataFrame([cand])
                        hdr = not os.path.exists(report_file)
                        df_cand.to_csv(report_file, mode="a", index=False, header=hdr)
                except Exception as e:
                    logger.debug("Worker error on TIC %s: %s", tic, e)

    logger.info("======================================================================")
    logger.info("COMPLETED TOI COMPANION SEARCH!")
    logger.info("Total Sibling Exoplanet Candidates Found: %d", len(total_candidates))
    logger.info("Summary Report: %s", report_file)
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
