#!/bin/bash
# ==============================================================================
# ALDEBARAN v9.0 - END-TO-END AUTOMATED CLUSTER LAUNCHER
# ==============================================================================
# Submits the 8-node screening array (Tier-1) and automatically schedules the
# shard merge and Pleiades deep vetting (Tier-2) to execute once all shards finish.
# Usage: ./cluster/submit_all.sh or cd cluster && ./submit_all.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "======================================================================"
echo "Submitting Aldebaran v9.0 End-to-End Pipeline to MetaCentrum"
echo "Repository Root: $REPO_DIR"
echo "======================================================================"

# 1. Submit Tier-1 array across 8 sectors
ARRAY_JOB_ID=$(qsub -J 1-8 cluster/run_aldebaran.pbs)
echo "[+] Step 1/2: Submitted Tier-1 array job -> ${ARRAY_JOB_ID}"

# 2. Submit Tier-2 reducer with afterany dependency on the entire array
MERGE_JOB_ID=$(qsub -W depend=afterany:"${ARRAY_JOB_ID}" cluster/run_merge.pbs)
echo "[+] Step 2/2: Submitted Tier-2 merge reducer job -> ${MERGE_JOB_ID}"

echo "======================================================================"
echo "Workflow successfully scheduled!"
echo "Tier-1 Array:        ${ARRAY_JOB_ID}"
echo "Tier-2 Auto-Reducer: ${MERGE_JOB_ID} (Status: Hold/Waiting for array)"
echo ""
BASE_ID="${ARRAY_JOB_ID%%\[\]*}"
echo "Monitor overall progress: qstat -u \$USER"
echo "Monitor individual array tasks: qstat -t ${BASE_ID}[]"
echo "======================================================================"
