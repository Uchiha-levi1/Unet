#!/bin/bash
# Submit the sharded preprocessing array, then finalize once all shards succeed. Run from the repo root:
#   bash jobs/preprocess_v3_submit.sh
# Resubmitting resumes: finished families (Data_v3/meta/<family>.json) are skipped.
set -euo pipefail
JID=$(sbatch --parsable jobs/preprocess_v3.sbatch)
FID=$(sbatch --parsable --dependency=afterok:$JID jobs/preprocess_v3_finalize.sbatch)
echo "array $JID, finalize $FID"
