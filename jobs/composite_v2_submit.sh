#!/bin/bash
# Submit the sharded compositing array, then finalize once all shards succeed. Run from the repo root:
#   bash jobs/composite_v2_submit.sh
# Resubmitting resumes: finished inspections (Data_v2/meta/<id>.json) are skipped.
set -euo pipefail
JID=$(sbatch --parsable jobs/composite_v2.sbatch)
FID=$(sbatch --parsable --dependency=afterok:$JID jobs/composite_v2_finalize.sbatch)
echo "array $JID, finalize $FID"
