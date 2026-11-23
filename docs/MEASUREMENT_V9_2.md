# Aldebaran 0.9.2: release measurement

This document reports the one measurement on which the 0.9.2 claims rest (internal ID M-RUN-RC2-01, run on MetaCentrum on 18 and 19 November 2026). It covers Tier 1 (`aldebaran.py`) at commit e474c65c2c with its default switches (A2, ARCHIVAL_COVERAGE, V_SHAPE_FLAG, DEPTH_UPPER). Tier 2 to 4 were not measured.

## How it was measured

- **Fresh hosts.** 1,126 TESS hosts were drawn for this run and had never been used to develop or tune the pipeline. The development cohort (S5) was kept separate by host.
- **Preregistration.** The cohorts, estimands, decision rules and the comparison with SHERLOCK were fixed and frozen before any detector ran. Nothing was rerun, dropped or topped up after an outcome was seen. Every departure from the plan is recorded as a deviation (see below).
- **Independent recomputation.** A second reviewer recomputed every decision-driving number from the per-case records, not from the summary tables, and found no discrepancy.
- **Scale.** 3,066 Aldebaran runs and 2,740 SHERLOCK runs, all with a final result (no unresolved errors).
- **Offline and blinded.** Stellar parameters came from the frozen cohort manifest, and the known-signal context was empty in every arm. M2 hosts were fully blinded: a planet found at a sibling's ephemeris counts as a miss of the target (7 M2 planets); the effect of production sibling masking is not measured. M3 therefore measures the photometric vetting alone, without catalogue masking.

Arms:

| Arm | What | Size |
|---|---|---|
| M1 | Planets injected into every real light curve of the host before detrending | 1,800 planned on 300 hosts, six SNR bins of 300; 60 could not be drawn inside the domain on their host, leaving 1,740 injections on 298 hosts |
| M2 | Known archival planets, blinded | 176 hosts, 250 planets, 199 in the declared domain |
| M3 | Known eclipsing binaries (Prša et al. 2022) and TFOPWG false positives | 100 + 50 hosts |
| M4 | Quiet stars without known signals: N1, the light curves scrambled in time (first orbit reversed, second kept forward, segments swapped, archival sectors reversed and shifted), so real periodic signals are destroyed but the noise is kept; N2, the same stars unmodified, an upper bound that can include real uncatalogued signals | 500 + 500 star-sectors |
| M5 | Ephemeris accuracy of recovered M1 injections | 886 |

Declared domain:
- host: CROWDSAP $\ge$ 0.70, RUWE $\le$ 1.4, $R_\star \le 1.75\,R_\odot$, $T_\text{eff} \le$ 9,500 K;
- signal: depth 180 to 30,000 ppm (no completeness claim below 180 ppm), $R_p \le 15\,R_\oplus$. The detector rejects $R_p < 0.8\,R_\oplus$ by design; this floor is not a domain bound, so such injections stay in the denominators. The DEPTH_UPPER gate accepts core depths up to 35,000 ppm, above the 30,000 ppm domain ceiling;
- injected geometry: circular orbits, impact parameter 0 to 0.8, period log-uniform from the larger of 1 day and the period at which $a/R_\star$ = 2.6, up to 13 days, at least 3 supported transits in the discovery sector. The injector was validated to 11.5 s in contact time and 2.4 × 10⁻⁴ relative in depth. The pixel injection uses no PRF model, so M1 does not certify centroid or pixel-gate completeness beyond that model;
- one discovery sector per star; any archival TESS sectors of the host were used for multi-epoch verification.

## Decisions fixed in advance

| Rule | Condition | Result | Outcome |
|---|---|---|---|
| D3, false alarms | N1 final candidates at SDE $\ge$ 6.0: one-sided 95 % upper bound $\le$ 1.5 % (at most 2 of 500) | 10 of 500, upper bound 3.37 % | **Failed** |
| D1, grey zone | completeness gain from SDE $\ge$ 7.5 to $\ge$ 6.0 at injected SNR 5 to 9 $\ge$ 5.0 points, and D3 passes | gain 5.86 points (34 of 580), D3 failed | **Claim withdrawn** |
| D2, preliminary label | known EBs and false positives ending as candidates $\le$ 10 % with upper bound $\le$ 15 % | 25 of 150 = 16.7 %, upper bound 22.5 % | **Label applies** |

Consequences for the release text: no grey-zone sensitivity claim, no false-alarm rate claimed at the production threshold, and the output is described as a preliminary transit candidate list that needs external secondary-eclipse or RV vetting.

## Results

**M1, injected planets** (final matched candidate; two-sided 95 % Clopper-Pearson intervals). M1 is conditioned on signals that can be drawn inside the declared domain on each host; a draw fails, for example, when every in-domain depth on a quiet host already exceeds the target SNR.

| Injected SNR | Planned | Not drawable | Evaluated | Matched candidate | Interval |
|---|---|---|---|---|---|
| 4 to 6 | 300 | 18 | 282 | 4.3 % | 2.2 to 7.3 % |
| 6 to 8 | 300 | 12 | 288 | 21.5 % | 16.9 to 26.7 % |
| 8 to 10 | 300 | 5 | 295 | 46.4 % | 40.6 to 52.3 % |
| 10 to 15 | 300 | 4 | 296 | 65.5 % | 59.8 to 70.9 % |
| 15 to 25 | 300 | 8 | 292 | 78.8 % | 73.6 to 83.3 % |
| 25 to 35 | 300 | 13 | 287 | 87.5 % | 83.1 to 91.1 % |
| All | 1,800 | 60 | 1,740 | 50.9 % | 48.6 to 53.3 % (host-cluster bootstrap over 298 hosts) |
| All, $R_p \ge 0.8\,R_\oplus$ | | | 1,738 | 51.0 % | 48.6 to 53.4 % (host-cluster bootstrap) |

Over all 1,800 planned injections, counting the 60 that could not be drawn as misses, the rate is 49.2 % (886 of 1,800).

Of the 854 injections lost, 634 were lost at the search stage (500 never produced a peak above SDE 6) and 216 in vetting, most often at the secondary-eclipse check (53), a vetting hold (51), the morphology check (45) and the archival check (33); 6 were rejected by the radius gate, and neither of the 2 injections with a true radius below 0.8 $R_\oplus$ was recovered. 4 more passed vetting but were not the reported candidate. Recovery falls with fewer transits in the sector (3 transits: 26.5 %; 10 or more: 64.8 %) and with long transits (T14 above 0.18 d: 23.2 %, against 52.5 % for shorter ones).

**M2, known planets:** 36 of 199 in-domain planets (18.1 %, 95 % interval 13.0 to 24.2 %) ended as a matched candidate. Eight of the 199 have no valid reference ephemeris and count as not recovered; over the other 191 the rate is 18.8 % (13.6 to 25.1 %). Over all 250 planets, in domain or not, 39 (15.6 %, 11.3 to 20.7 %). By catalogue SNR: below 10, 2 of 90; 10 to 25, 16 of 34; 25 or more, 18 of 35. Planets with at least 3 observed transits in the sector: 36 of 84 (42.9 %); with at least 3 transits and catalogue SNR $\ge$ 10: 34 of 47 (72.3 %). Planets with transits longer than 0.18 d were almost never recovered (2 of 67).

**M3, eclipsing binaries and false positives:** 25 of 150 ended as candidates (16.7 %, two-sided 95 % interval 11.1 to 23.6 %, one-sided upper bound 22.5 %). Prša et al. EBs: 6 of 100 (6.0 %, 2.2 to 12.6 %, upper bound 11.5 %). TFOPWG false positives: 19 of 50 (38.0 %, 24.7 to 52.8 %, upper bound 50.6 %). Of the 25, 22 were reported at the catalogue period and 3 at an alias. None of the 25 has a measured core depth between 30,000 and 35,000 ppm, the band that only the DEPTH_UPPER gate admits. Development disclosure: when DEPTH_UPPER was calibrated on the S5 development cohort, one binary outside the signal domain, TIC 239262664 (1 of 86 in that stratum), ended as a candidate. It is an equal-depth binary detected at half its period, with a measured core depth of 34,287 ppm; its odd and even eclipses differ too little for the odd/even check to reject it.

**M4, quiet stars:** N1 (time-scrambled), 10 of 500 final candidates (2.0 %, one-sided 95 % upper bound 3.37 %), all from a single sector, all with SDE 6.50 to 7.39; none at SDE $\ge$ 7.5 (observed after the fact, not a tested threshold). N2 (unmodified), 3 of 500 (0.6 %, upper bound 1.54 %), also all single-sector with SDE below 7.5. At the search stage, before vetting, 55 of the 500 N1 stars had a top peak at SDE $\ge$ 6.0.

**M5, ephemerides:** over 886 recovered injections, the period error has median 0.00001 d and central 68 % from −0.0013 to +0.0009 d; the T0 error has median −0.00017 d and central 68 % from −0.0046 to +0.0048 d. Measured depth exceeds the true depth by a median factor of 1.11, the radius by 1.03. Three limits apply to these numbers:
- the reported period and T0 uncertainties are heuristics from the discovery sector (Carter et al.), even for refined ephemerides; they exist for only 30 of the 886 candidates, and all 30 cover the truth at 1, 2 and 3 sigma;
- the sample is the matched candidates only, so at low SNR it keeps the injections that happened to be measured well;
- the radius ratio uses the catalogue stellar radius and leaves out its error.

## Comparison with SHERLOCK

SHERLOCK 1.6.1 ran on the same injected light curves and the same quiet stars, with periods 0.8 to 13 days, up to 3 runs per star and no Watson vetting. The comparison is at the search stage: Aldebaran counts a FUNDAMENTAL match among its top 3 BLS peaks at SDE $\ge$ 6.0, SHERLOCK any accepted (score 1) FUNDAMENTAL row. Differences are Aldebaran minus SHERLOCK on the same injection or star.

| Row | n | Aldebaran | SHERLOCK | Difference | Paired test |
|---|---|---|---|---|---|
| SNR 4 to 6 | 282 | 9.2 % | 17.4 % | −8.2 points | McNemar p = 0.0001 |
| SNR 6 to 8 | 288 | 30.6 % | 45.8 % | −15.3 points | p < 0.0001 |
| SNR 8 to 10 | 295 | 63.1 % | 75.9 % | −12.9 points | p < 0.0001 |
| SNR 10 to 15 | 296 | 83.4 % | 87.5 % | −4.1 points | p = 0.11 |
| SNR 15 to 25 | 292 | 94.9 % | 95.9 % | −1.0 points | p = 0.66 |
| SNR 25 to 35 | 287 | 98.3 % | 97.9 % | +0.3 points | p = 1 |
| SNR 5 to 10 | 728 | 40.7 % | 53.7 % | −13.0 points | host bootstrap −16.8 to −9.5 |
| SNR 10 to 25 | 588 | 89.1 % | 91.7 % | −2.6 points | host bootstrap −5.4 to +0.3 |
| SNR 10 and above | 875 | 92.1 % | 93.7 % | −1.6 points | host bootstrap −3.8 to +0.6 |
| Quiet stars N1 (alarms) | 500 | 11.0 % | 39.0 % | −28.0 points | p < 0.0001 |
| Quiet stars N2 (alarms) | 500 | 12.6 % | 36.8 % | −24.2 points | p < 0.0001 |

SHERLOCK finds more injected planets below SNR 10 and raises far more alarms on quiet stars. From SNR 10 up, the two search stages do not differ beyond the intervals. After vetting, Aldebaran keeps 10 of the 500 N1 alarms; SHERLOCK has no comparable vetting in this setup, and even its SDE $\ge$ 7.5 subset flags 18.2 % of the N1 stars. The pooled rows share hosts across SNR bins, so they carry a host-cluster bootstrap interval instead of a McNemar test. The comparison decides nothing.

## Deviations from the plan

None of these changes an estimand, a denominator or a decision rule.
1. The run manifest was converted to the driver's schema (two keys); every other key was checked equal.
2. A first preflight attempt failed on a defect in a new environment check and is kept; the second passed.
3. The detector ran as one serial job instead of array jobs (wall time only).
4. Injected light-curve copies were not kept; SHERLOCK regenerated each one and required the same SHA256 the detector had recorded.
5. SHERLOCK ran alongside the detector instead of after it, to save a day; no outcome of either was read before both had finished.
6. Four SHERLOCK runs stalled on network storage before producing anything and were rerun once.
7. Regenerating the injected inputs gave byte-different files on newer AMD CPUs (Zen 3 and Zen 4). The check refused every such input, and those cases were rerun on the detector's own CPU type, where all 2,240 regenerated inputs (1,740 injected, 500 scrambled) matched; the 500 unmodified N2 inputs also passed the SHA256 check.
8. The frozen SHERLOCK adapter crashed while scoring 29 injected cases whose candidate list ended with a row without a period. A guard that labels such rows as having no ephemeris was added. It gave identical scoring on all 1,711 cases that had finished, and on a deliberately broken check it detected the change. The 29 cases were rerun once with the guard.

## Limits

- One measurement on one cohort. The intervals above are its statistical uncertainty; they do not cover a different host population, other sectors or other cadences.
- Mostly single-sector stars. The injected and quiet hosts came from a cohort frame that kept few archival observations: 289 of the 300 M1 hosts and 473 of the 500 M4 hosts had only their discovery sector. The M1 and M4 numbers therefore describe single-sector stars, and the archival false-pass rate with period refinement on quiet stars had no trigger at all and is not measured. M2 and M3 hosts were queried per host and mostly have several sectors.
- Tier 1 only. Tier 2 to 4 outputs (VPS, TRICERATOPS FPP, archival forensics) are uncalibrated, and their attrition, including the Pleiades `is_clean` and VPS cuts, is not measured. Provided they only remove Tier 1 candidates, Tier 1 overstates end-to-end recovery (M1, M2) and end-to-end false rates (M3, M4).
- Purity, the fraction of real planets among the candidates, is not given. It would need a calibrated false-alarm rate, which this measurement failed to establish; a false-positive prior for the searched population, which the fixed M3 mix of binaries and false positives is not; and the occurrence of detectable planets among the searched stars, which was not measured.
- The grey zone ($6.0 \le$ SDE $\le$ 7.5) is the target of the next development cycle, which needs a new fresh cohort to test any new rule.
