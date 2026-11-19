"""Offline M harness for the v9.2 measurement (roadmap S4.1, S4.3, S4.4, S4.5 preflight; TASK S4-M-PACKAGE-01).

The harness measures nothing by itself. It prepares inputs (nulls, injections), routes local products into the
production entry points, and turns the production records into the M tables. Every decision comes from the
production code; the harness holds no detection threshold of its own.

Modules:
  stats        exact binomial bounds, fixed order statistics, host-cluster bootstrap, sample sizes
  nulls        N1 and N1-Archival null constructions (ARGON_DECISIONS item 31.7), applied to product files
  fidelity     development fidelity gate (item 31.7 as amended by item 32) on light curve arrays
  transit_model geometry, exact-orbit reference model, the item 31.5 model check (S4.4)
  injection    SNR_inj, joint solve, M1 draws on the declared bins, light curve and pixel injection
  matching     frozen event-matching rules (bank_valid_01 scorer)
  routing      offline product routing into aldebaran.analyze_single_target, and the S4.3 routing controls
  estimands    per-stage attrition ledger and the M1 to M5, grey-zone and item 31 diagnostic calculators
  preflight    C14 preflight over cohort metadata (AGENTS.md "Preflight before long runs")
  runner       cohort runner with the S3.6 bound resume, and the fixture dry run
  fixtures     synthetic SPOC-format products for tests and the dry run (engineering only)
"""

HARNESS_VERSION = "m_harness.v1"
