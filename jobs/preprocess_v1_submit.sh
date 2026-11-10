#!/bin/bash
# Submit the sharded preprocessing array, then finalize once all shards succeed. Run from the repo root:
#   bash jobs/preprocess_v1_submit.sh
# Resubmitting resumes: finished inspections (Data_v1/meta/<id>.json) are skipped.
set -euo pipefail
JID=$(sbatch --parsable jobs/preprocess_v1.sbatch)
FID=$(sbatch --parsable --dependency=afterok:$JID jobs/preprocess_v1_finalize.sbatch)
echo "array $JID, finalize $FID"
