#!/usr/bin/env python3
"""
ALDEBARAN EXOPLANET SEARCH PIPELINE â€” MODULE: TIER-4 TRICERATOPS BAYESIAN FPP/NFPP
Computes Bayesian False Positive Probability (FPP) and Nearby False Positive Probability
(NFPP) estimates conditional on the supplied data, astrophysical scenarios, and model priors.
Optimized with Phase-Folded Adaptive Binning to prevent OOM on multi-sector datasets.
"""

import sys
import os
import argparse
import json
from importlib.metadata import PackageNotFoundError, version
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ephemeris_contract import BTJD_OFFSET, bjd_to_btjd, fold_phase_centered, hours_to_days

SUPPORTED_TRICERATOPS_VERSION = "1.0.21"
TRICERATOPS_EXPTIME_DAYS = 0.00139
# Lightkurve 2.6 SearchResult exptime is in seconds. Keep the library
# default exposure (rounded 0.00139 d) unchanged; accept nominal 120-s products only.
SPOC_EXPOSURE_SECONDS = 120.0


def _pipeline_aperture_pixels(tpf):
    mask = np.asarray(tpf.pipeline_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("SPOC TPF pipeline aperture is empty or not two-dimensional.")
    rows, columns = np.nonzero(mask)
    return np.column_stack((int(tpf.column) + columns, int(tpf.row) + rows))


def _sap_flux_only(flux, flux_column):
    flux = np.asarray(flux, dtype=float)
    if flux_column == "sap_flux":
        return flux
    raise ValueError(
        "TRICERATOPS 1.0.21 requires SAP flux for this adapter. "
        "PDCSAP conversion is not enabled without an approved scientific contract."
    )


def _download_spoc_sector(lk, tic_id, sector, flux_column):
    if flux_column != "sap_flux":
        raise ValueError(
            "SPOC SAP flux is required; PDCSAP was requested and no automatic "
            "product substitution or CROWDSAP conversion is allowed."
        )
    search_args = {
        "target": f"TIC {tic_id}",
        "mission": "TESS",
        "author": "SPOC",
        "sector": int(sector),
    }
    lc_search = lk.search_lightcurve(**search_args)
    tpf_search = lk.search_targetpixelfile(**search_args)
    if len(lc_search) == 0 or len(tpf_search) == 0:
        raise ValueError(f"No paired SPOC light-curve and TPF products for sector {sector}.")

    lc_exptime = np.asarray(lc_search.table["exptime"], dtype=float)
    tpf_exptime = np.asarray(tpf_search.table["exptime"], dtype=float)
    matches = [
        (i, j)
        for i, lc_time in enumerate(lc_exptime)
        for j, tpf_time in enumerate(tpf_exptime)
        if np.isclose(lc_time, tpf_time, rtol=0, atol=1e-10)
    ]
    if not matches:
        raise ValueError(f"No cadence-matched SPOC light-curve and TPF products for sector {sector}.")

    best_matches = [(i, j) for i, j in matches
                    if lc_exptime[i] == SPOC_EXPOSURE_SECONDS]
    if not best_matches:
        raise ValueError(f"No supported 120-second SPOC pair for sector {sector}; "
                         "the unchanged TRICERATOPS exposure default requires this cadence.")
    if len(best_matches) != 1:
        raise ValueError(f"Ambiguous cadence-matched SPOC products for sector {sector}.")

    lc_index, tpf_index = best_matches[0]
    try:
        lc = lc_search[lc_index].download(flux_column="sap_flux")
    except Exception as e:
        raise RuntimeError(
            f"SPOC SAP flux is unavailable for sector {sector}; no PDCSAP "
            f"fallback was attempted: {e}"
        ) from e
    tpf = tpf_search[tpf_index].download()
    lc = lc.remove_nans().remove_outliers(sigma=5)
    if int(tpf.sector) != int(sector):
        raise ValueError(f"Downloaded TPF sector {tpf.sector} does not match requested sector {sector}.")

    if int(lc.sector) != int(sector):
        raise ValueError(f"Downloaded light-curve sector {lc.sector} does not match requested sector {sector}.")

    flux = _sap_flux_only(lc.flux.value, flux_column)
    flux_err = np.asarray(lc.flux_err.value, dtype=float)
    return np.asarray(lc.time.btjd, dtype=float), flux, flux_err, _pipeline_aperture_pixels(tpf)


def _normalize_sap_sector(times, flux, errors, t0_btjd, period, dur_days):
    """Scale each aperture product before combining; preserve errors in the same units."""
    times, flux, errors = (np.asarray(x, dtype=float) for x in (times, flux, errors))
    if times.ndim != 1 or flux.shape != times.shape or errors.shape != times.shape:
        raise ValueError("SAP time/flux/error arrays must be aligned one-dimensional arrays.")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(flux))
            and np.all(np.isfinite(errors)) and np.all(errors >= 0)):
        raise ValueError("SAP time/flux/error arrays contain invalid evidence.")
    dt = fold_phase_centered(times, t0_btjd, period, in_days=False) * period
    # Reuse the existing profile's out-of-transit window, before sector combination.
    oot = (np.abs(dt) > 0.6 * dur_days) & (np.abs(dt) <= 2.0 * dur_days)
    if not np.any(oot):
        raise ValueError("SAP sector has no out-of-transit normalization support.")
    baseline = float(np.median(flux[oot]))
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("SAP sector has no positive finite reference flux.")
    return flux / baseline, errors / baseline, baseline


def run_triceratops_validation(tic_id, period, t0_btjd, duration_hours, depth_ppm, sectors, n_mc=5, n_bins=60, results_dir: str = None, flux_column: str = "sap_flux"):
    print(f"\n{'='*75}")
    print(f" ALDEBARAN TIER-4 BAYESIAN FPP/NFPP ASSESSMENT (TRICERATOPS)")
    print(f" Target: TIC {tic_id}")
    print(f" Ephemeris: P = {period:.6f} d | T0 = {t0_btjd:.5f} BTJD | Depth = {depth_ppm:.1f} ppm")
    print(f" Sectors: {sectors} | Target Phase Bins: {n_bins} | MC Draws: {n_mc}")
    print(f"{'='*75}\n")

    try:
        installed_version = version("triceratops")
        if installed_version != SUPPORTED_TRICERATOPS_VERSION:
            raise RuntimeError(
                f"This adapter requires TRICERATOPS {SUPPORTED_TRICERATOPS_VERSION}; "
                f"found {installed_version}."
            )
        import triceratops.triceratops as tr
    except PackageNotFoundError as e:
        print(f"[!] TRICERATOPS import error: {e}")
        return False
    except ImportError as e:
        print(f"[!] TRICERATOPS import error: {e}")
        return False
    except RuntimeError as e:
        print(f"[!] Unsupported TRICERATOPS version: {e}")
        return False
    if flux_column != "sap_flux":
        print(
            "[!] This TRICERATOPS 1.0.21 adapter requires SPOC SAP flux. "
            "PDCSAP conversion is not enabled without an approved scientific contract."
        )
        return False

    dur_days = hours_to_days(duration_hours)

    print(f"[TRICERATOPS] Initializing target object for TIC {tic_id}...")
    try:
        target = tr.target(ID=int(tic_id), sectors=np.array(sectors, dtype=int))
        target.sectors = np.array(target.sectors, dtype=int)
        star0 = target.stars.iloc[0]
        r_star = star0.get('rad', np.nan)
        teff = star0.get('Teff', np.nan)
        print(f"   -> Target host properties: R_* = {r_star:.3f} R_sun, T_eff = {teff:.0f} K" if np.isfinite(r_star) else f"   -> Target loaded.")
        print(f"   -> Stars in aperture catalog: {len(target.stars)} stars loaded.")
    except Exception as e:
        print(f"[!] Error initializing TRICERATOPS target: {e}")
        return False

    # Fetch explicitly paired SAP/TPF products; never stitch raw aperture counts.
    import lightkurve as lk
    sector_normalization = []
    try:
        if sorted(target.sectors.tolist()) != sorted(int(s) for s in sectors):
            raise ValueError("TRICERATOPS target sectors differ from requested evidence.")
        time_parts, flux_parts, error_parts, aperture_pixels = [], [], [], []
        for sector in target.sectors:
            times, flux, errors, pixels = _download_spoc_sector(lk, tic_id, int(sector), flux_column)
            flux, errors, baseline = _normalize_sap_sector(
                times, flux, errors, t0_btjd, period, dur_days)
            time_parts.append(times)
            flux_parts.append(flux)
            error_parts.append(errors)
            aperture_pixels.append(pixels)
            sector_normalization.append({"sector": int(sector), "reference_flux": baseline})
        time_all = np.concatenate(time_parts)
        f_raw, e_raw = np.concatenate(flux_parts), np.concatenate(error_parts)
        phase = fold_phase_centered(time_all, t0_btjd, period, in_days=False)
        dt_days = phase * period

        # Crop to transit window: [-2.0 * duration, +2.0 * duration]
        win_mask = np.abs(dt_days) <= (2.0 * dur_days)
        dt_win = dt_days[win_mask]
        f_win = f_raw[win_mask]
        e_win = e_raw[win_mask]
        print(f"   -> Raw multi-sector cadences in transit window: {len(dt_win)} points.")
        
        # CRITICAL OPTIMIZATION: Phase binning to prevent 10.8 GB MemoryError
        # 132,169 simulated scenarios x 10,927 raw points = 10.8 GB array allocation.
        # Binning into n_bins reduces allocation to ~60 MB (<0.6% of original RAM)
        # while increasing SNR per data point by sqrt(N_transits) = sqrt(94) ~ 9.7x!
        bins = np.linspace(-2.0 * dur_days, 2.0 * dur_days, n_bins + 1)
        bin_cents = 0.5 * (bins[:-1] + bins[1:])
        b_time, b_flux, b_err = [], [], []
        
        for i in range(n_bins):
            m_bin = (dt_win >= bins[i]) & (dt_win < bins[i+1])
            n_pts = np.sum(m_bin)
            if n_pts >= 3:
                med_val = float(np.nanmedian(f_win[m_bin]))
                sem_val = float(np.nanstd(f_win[m_bin]) / np.sqrt(n_pts)) if n_pts > 1 else float(np.nanmedian(e_win[m_bin]))
                b_time.append(bin_cents[i])
                b_flux.append(med_val)
                b_err.append(sem_val)
                
        time_binned = np.array(b_time, dtype=float)
        flux_binned = np.array(b_flux, dtype=float)
        
        # Normalize out-of-transit flux to 1.0
        oot_mask = np.abs(time_binned) > (0.6 * dur_days)
        norm_factor = float(np.nanmedian(flux_binned[oot_mask])) if np.sum(oot_mask) >= 5 else float(np.nanmedian(flux_binned))
        flux_binned = flux_binned / norm_factor
        
        # Retain the pending adapter estimator exactly; it is not S3's 90%-box mean.
        # This diagnostic convention still needs scientific review before real-target use.
        transit_mask = np.abs(time_binned) <= (0.5 * dur_days)
        if not np.any(transit_mask):
            raise ValueError("No populated phase bins fall inside the transit duration.")
        sap_depth = float(1.0 - np.nanmedian(flux_binned[transit_mask]))
        if not np.isfinite(sap_depth) or sap_depth <= 0:
            raise ValueError(f"Measured SAP profile depth is not positive: {sap_depth}.")
        input_contract = {
            "flux_column": flux_column, "depth_fraction": sap_depth,
            "depth_estimator": "median_binned_full_duration_provisional",
            "depth_policy_status": "SCIENTIFIC_REVIEW_REQUIRED",
            "reference_depth_ppm": float(depth_ppm),
            "sector_normalization": sector_normalization,
            "product_exposure_seconds": SPOC_EXPOSURE_SECONDS,
            "library_default_exposure_days": TRICERATOPS_EXPTIME_DAYS,
        }
        target.calc_depths(tdepth=sap_depth, all_ap_pixels=aperture_pixels)
        print(f"   -> SAP depth {sap_depth * 1e6:.1f} ppm; reference input {depth_ppm:.1f} ppm. "
              "Provisional depth estimator; not scientific approval.")

        # Representative flux uncertainty (scalar float for TRICERATOPS)
        flux_err_scalar = float(np.nanmedian(b_err) / norm_factor)
        
        print(f"   -> Phase-folded binned profile: {len(time_binned)} points (RAM footprint: ~{132169 * len(time_binned) * 8 / (1024**2):.1f} MB)")
        print(f"   -> Effective binned photometric error: {flux_err_scalar*1e6:.1f} ppm")
        
    except Exception as e:
        print(f"[!] Warning fetching and binning light curve: {e}")
        return False

    print(f"\n[TRICERATOPS] Calculating Bayesian scenario probabilities (N_MC={n_mc} draws)...")
    fpps = []
    nfpps = []
    probs_history = []
    for i in range(n_mc):
        try:
            target.calc_probs(
                time=time_binned,
                flux_0=flux_binned,
                flux_err_0=flux_err_scalar,
                P_orb=period,
                parallel=True,
                verbose=0
            )
            fpps.append(target.FPP)
            nfpps.append(target.NFPP)
            probs_history.append(target.probs.copy())
            print(f"   [Iter {i+1:2d}/{n_mc}] FPP = {target.FPP:.5f} ({target.FPP*100:.2f}%) | NFPP = {target.NFPP:.5f} ({target.NFPP*100:.2f}%)")
        except Exception as e:
            print(f"   [Iter {i+1:2d}] Failed: {e}")
            continue

    if not fpps:
        print("[!] All Monte Carlo iterations failed.")
        return False

    mean_fpp = float(np.mean(fpps))
    std_fpp = float(np.std(fpps))
    mean_nfpp = float(np.mean(nfpps))
    std_nfpp = float(np.std(nfpps))

    print(f"\n{'='*65}")
    print(f" BAYESIAN FPP/NFPP ASSESSMENT (TRICERATOPS)")
    print(f" Conditional scenario-model FPP: {mean_fpp:.4f} Â± {std_fpp:.4f} ({mean_fpp*100:.2f}%)")
    print(f" Conditional nearby false-positive probability (NFPP): {mean_nfpp:.4f} Â± {std_nfpp:.4f} ({mean_nfpp*100:.2f}%)")
    
    # These thresholds classify conditional scenario-model outputs; they are not independent validation.
    if mean_fpp < 0.015 and mean_nfpp < 0.001:
        disposition = "EXPLORATORY CONDITIONAL SCENARIO-MODEL THRESHOLD PASS (FPP < 0.015, NFPP < 0.001; not independent validation)"
    elif mean_fpp < 0.05:
        disposition = "VERY LOW CONDITIONAL FPP CANDIDATE (TRICERATOPS FPP < 5%)"
    elif mean_fpp < 0.10:
        disposition = "LOW CONDITIONAL FPP CANDIDATE (TRICERATOPS FPP < 10%)"
    elif mean_fpp < 0.50:
        disposition = "MODERATE CONDITIONAL FPP CANDIDATE (TRICERATOPS FPP < 50%)"
    else:
        disposition = "TRICERATOPS MODEL FAVORS FALSE-POSITIVE / BLEND SCENARIOS (conditional FPP >= 50%)"
    print(f" Scientific Disposition: {disposition}")
    print(f"{'='*65}\n")

    if results_dir is None:
        results_dir = "tess_results"
    out_dir = os.path.join(results_dir, "triceratops")
    os.makedirs(out_dir, exist_ok=True)

    # Plot field and scenario fits
    try:
        fits_base = os.path.join(out_dir, f"triceratops_fits_TIC_{tic_id}")
        target.plot_fits(time=time_binned, flux_0=flux_binned, flux_err_0=flux_err_scalar, save=True, fname=fits_base)
        plt.savefig(fits_base + ".png", dpi=120)
        plt.close('all')
        print(f"[TRICERATOPS] Saved scenario fits plot to {fits_base}.png/.pdf")
    except Exception as e:
        print(f"[!] Notice plotting fits: {e}")

    try:
        field_base = os.path.join(out_dir, f"triceratops_field_TIC_{tic_id}")
        sec_to_plot = sectors[0] if sectors else None
        target.plot_field(sector=sec_to_plot, save=True, fname=field_base)
        plt.savefig(field_base + ".png", dpi=120)
        plt.close('all')
        print(f"[TRICERATOPS] Saved aperture field plot to {field_base}.png/.pdf")
    except Exception as e:
        print(f"[!] Notice plotting field: {e}")

    # Save CSV: compute mean probabilities across all Monte Carlo draws
    base_df = target.probs.copy()
    if probs_history:
        all_p = np.array([p['prob'].values for p in probs_history])
        mean_p = np.mean(all_p, axis=0)
        base_df['prob'] = mean_p
        if base_df['prob'].sum() > 0:
            base_df['prob'] = base_df['prob'] / base_df['prob'].sum()
    prob_df = base_df
    prob_df["input_contract_json"] = json.dumps(input_contract, sort_keys=True)

    csv_path = os.path.join(out_dir, f"triceratops_probs_TIC_{tic_id}.csv")
    prob_df.to_csv(csv_path)
    print(f"[TRICERATOPS] Saved scenario probabilities table (averaged over {len(probs_history)} MC draws) to {csv_path}")

    # Print scenario breakdown
    print(f"\n[TRICERATOPS] Top 5 Most Probable Scenarios (Averaged across {len(probs_history)} MC draws):")
    sorted_probs = prob_df.sort_values(by="prob", ascending=False)
    for idx, r in sorted_probs.head(5).iterrows():
        print(f"   {r['scenario']:<10}: P = {r['prob']:.4f} ({r['prob']*100:.2f}%)")

    # Astrophysical accounting breakdown
    p_tp = prob_df[prob_df['scenario'] == 'TP']['prob'].sum()
    p_ptp = prob_df[prob_df['scenario'] == 'PTP']['prob'].sum()
    p_dtp = prob_df[prob_df['scenario'] == 'DTP']['prob'].sum()
    p_stp = prob_df[prob_df['scenario'] == 'STP']['prob'].sum()
    p_primary_planet = p_tp + p_ptp + p_dtp
    p_bound_planet = p_primary_planet + p_stp
    p_eb_total = 1.0 - p_bound_planet
    print(f"\n[TRICERATOPS] Astrophysical Planet vs. False Positive Budget:")
    print(f"   -> Primary Host Planet (1 - FPP)  : {p_primary_planet*100:.2f}% (TP={p_tp*100:.2f}%, PTP={p_ptp*100:.2f}%, DTP={p_dtp*100:.2f}%)")
    print(f"   -> Secondary Companion Planet(STP): {p_stp*100:.2f}% (classified under formal FPP in TRICERATOPS)")
    print(f"   -> Total Bound Planet Probability : {p_bound_planet*100:.2f}%")
    print(f"   -> Non-Planetary EB Probability   : {p_eb_total*100:.2f}% (Nearby NEB={mean_nfpp*100:.2f}%)")

    return {
        'input_contract': input_contract,
        'tic_id': tic_id,
        'fpp_mean': mean_fpp,
        'fpp_std': std_fpp,
        'nfpp_mean': mean_nfpp,
        'nfpp_std': std_nfpp,
        'disposition': disposition
    }


def _sector_evidence(value):
    import re
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not re.fullmatch(r"[1-9][0-9]*(?:\s*,\s*[1-9][0-9]*)*", text):
        raise ValueError(f"Invalid explicit sector evidence: {value!r}")
    values = [int(x.strip()) for x in text.split(",")]
    if len(set(values)) != len(values):
        raise ValueError("Duplicate sectors in explicit evidence")
    return sorted(values)


def validate_auto(n_mc: int = 5, n_bins: int = 60, top_csv: str = None, results_dir: str = None):
    """
    Automatically validates top CTOI candidates from exofop_ctoi_upload_top3.csv
    or prioritized_targets_ranking.csv.
    """
    if results_dir is None:
        results_dir = "tess_results"
    if top_csv is None:
        top_csv = os.path.join(results_dir, "exofop_ctoi_upload_top3.csv")
    if not os.path.exists(top_csv):
        print(f"[!] Top candidates table '{top_csv}' not found. Skipping auto-validation.")
        return

    with open(top_csv, encoding="utf-8") as stream:
        sep = "|" if "|" in stream.readline() else ","
    df_top = pd.read_csv(top_csv, sep=sep)
    col_tic = [c for c in df_top.columns if "TIC" in c.upper()][0]
    col_p = [c for c in df_top.columns if "PERIOD" in c.upper()][0]
    col_epoch = [c for c in df_top.columns if "EPOCH" in c.upper() or "T0" in c.upper()][0]
    col_depth = next((c for c in ("Depth", "measured_box_depth_ppm", "depth_ppm")
                      if c in df_top.columns), None)
    if col_depth is None:
        raise ValueError("No explicit photometric depth column; geometric/model aliases are not accepted")
    col_dur = [c for c in df_top.columns if "DUR" in c.upper()][0]

    prefix = "antares_" if os.path.basename(top_csv).startswith("antares_") else ""
    ranking_csv = os.path.join(results_dir, prefix + "prioritized_targets_ranking.csv")
    df_rank = pd.read_csv(ranking_csv) if os.path.exists(ranking_csv) else None

    targets_to_run = []
    for _, row in df_top.iterrows():
        tid = int(row[col_tic])
        p_val = float(row[col_p])
        epoch_val = float(row[col_epoch])
        t0_btjd = bjd_to_btjd(epoch_val)
        depth_val = float(row[col_depth])
        dur_val = float(row[col_dur])

        sector_sources = []
        if df_rank is not None and "tic_id" in df_rank.columns:
            matching = df_rank[pd.to_numeric(df_rank["tic_id"], errors="coerce") == tid]
            if "sectors" in matching.columns:
                sector_sources.extend(_sector_evidence(v) for v in matching["sectors"])
        if "sectors" in row.index:
            sector_sources.append(_sector_evidence(row["sectors"]))
        if not sector_sources:
            raise ValueError(f"No explicit sector provenance for TIC {tid} in {ranking_csv}")
        if any(value != sector_sources[0] for value in sector_sources[1:]):
            raise ValueError(f"Conflicting sector provenance for TIC {tid}")
        sectors = sector_sources[0]

        targets_to_run.append({
            "tic_id": tid,
            "period": p_val,
            "t0_btjd": t0_btjd,
            "duration_hours": dur_val,
            "depth_ppm": depth_val,
            "sectors": sectors
        })

    print(f"\n[+] Automatic TRICERATOPS validation queue: {len(targets_to_run)} targets loaded.")
    for cand in targets_to_run:
        run_triceratops_validation(
            tic_id=cand["tic_id"],
            period=cand["period"],
            t0_btjd=cand["t0_btjd"],
            duration_hours=cand["duration_hours"],
            depth_ppm=cand["depth_ppm"],
            sectors=cand["sectors"],
            n_mc=n_mc,
            n_bins=n_bins,
            results_dir=results_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Report Bayesian TRICERATOPS FPP/NFPP estimates conditional on scenarios and model priors.")
    parser.add_argument("--auto", action="store_true", help="Automatically validate all targets from top3 CSV")
    parser.add_argument("--top-csv", type=str, default=None, help="Explicit path to top candidates table")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory for ranking lookup and output files")
    parser.add_argument("--tic", type=int, default=350190949, help="TIC ID")
    parser.add_argument("--period", type=float, default=0.926120, help="Period in days")
    parser.add_argument("--t0", type=float, default=1325.61416, help="T0 BTJD")
    parser.add_argument("--duration", type=float, default=0.81, help="Duration in hours")
    parser.add_argument("--depth", type=float, default=1346.9, help="Depth in ppm")
    parser.add_argument("--sectors", type=int, nargs="+", default=[1, 28, 104, 105], help="TESS sectors")
    parser.add_argument("--mc", type=int, default=5, help="Number of Monte Carlo iterations")
    parser.add_argument("--bins", type=int, default=60, help="Number of phase bins across transit window")
    
    args = parser.parse_args()
    if args.auto:
        validate_auto(n_mc=args.mc, n_bins=args.bins, top_csv=args.top_csv, results_dir=args.results_dir)
    else:
        run_triceratops_validation(
            tic_id=args.tic,
            period=args.period,
            t0_btjd=args.t0,
            duration_hours=args.duration,
            depth_ppm=args.depth,
            sectors=args.sectors,
            n_mc=args.mc,
            n_bins=args.bins,
            results_dir=args.results_dir
        )
