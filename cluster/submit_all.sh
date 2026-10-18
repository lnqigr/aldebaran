#!/bin/bash
# Explicit scan-only submission. Merge requires a separately verified manifest.
set -euo pipefail
usage() { echo "Usage: $0 --scan-only [--array 1-8]"; }
fail() { echo "ERROR: $*" >&2; exit 2; }
[[ $# -gt 0 ]] || { usage >&2; exit 2; }
[[ "$1" == "--help" && $# -eq 1 ]] && { usage; exit 0; }
[[ "$1" == "--scan-only" ]] || fail "Only --scan-only is supported; automatic merge is unavailable."
shift
array="1-8"
if [[ $# -gt 0 ]]; then
    [[ $# -eq 2 && "$1" == "--array" ]] || fail "Unsupported or ambiguous options."
    array="$2"
fi
# run_aldebaran.pbs maps array indices onto eight configured sectors.
[[ "$array" =~ ^([1-8])(-([1-8]))?$ ]] || fail "Array must be an index or ascending range within 1-8."
if [[ -n "${BASH_REMATCH[3]:-}" ]]; then
    [[ "${BASH_REMATCH[1]}" -le "${BASH_REMATCH[3]}" ]] || fail "Descending array range."
fi
command -v qsub >/dev/null 2>&1 || fail "qsub is unavailable."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if ! job_id=$(qsub -J "$array" cluster/run_aldebaran.pbs); then
    fail "Scan submission failed."
fi
[[ "$job_id" =~ ^[0-9]+(\[\])?\.[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || fail "qsub returned an invalid job ID; inspect scheduler state before retrying."
echo "Scan array submitted: $job_id"
echo "Merge requires an independently checked explicit manifest and a separate submission."
