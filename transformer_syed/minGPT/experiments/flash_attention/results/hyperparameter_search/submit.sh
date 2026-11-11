#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
export MQP_SEARCH_DIR="$study_dir"
export MQP_SEARCH_OUT="${1:-$study_dir/run_fp32}"
mkdir -p "$MQP_SEARCH_OUT/logs"
if [[ -e "$MQP_SEARCH_OUT/jobs.tsv" ]]; then
    echo "Existing submission found at $MQP_SEARCH_OUT/jobs.tsv; choose a new directory." >&2
    exit 1
fi
python "$study_dir/study.py" plan --out "$MQP_SEARCH_OUT"
submit() {
    local stage="$1"
    shift
    local job
    job=$(sbatch --parsable --output="$MQP_SEARCH_OUT/logs/$stage-%A_%a.log" "$@" "$study_dir/run_gpu.sh" "$stage")
    job="${job%%;*}"
    printf '%s\t%s\n' "$stage" "$job" >> "$MQP_SEARCH_OUT/jobs.tsv"
    printf '%s' "$job"
}
smoke=$(submit smoke --time=00:15:00)
screen=$(submit screen --array=0-2%2 --dependency="afterok:$smoke")
refine=$(submit refine --dependency="afterok:$screen")
regularize=$(submit regularize --dependency="afterok:$refine")
batch=$(submit batch --dependency="afterok:$regularize" --time=02:00:00 --qos=long)
confirm=$(submit confirm --dependency="afterok:$batch" --time=02:00:00 --qos=long)
systems=$(submit systems --array=0-2%2 --dependency="afterok:$confirm" --time=04:00:00 --qos=long)
report_job=$(submit report --dependency="afterok:$systems" --time=00:10:00)
analysis_job=$(sbatch --parsable --dependency="afterok:$report_job" --output="$MQP_SEARCH_OUT/logs/analysis-%j.log" "$study_dir/finalize_gpu.sh")
printf 'analysis\t%s\n' "${analysis_job%%;*}" >> "$MQP_SEARCH_OUT/jobs.tsv"
cat "$MQP_SEARCH_OUT/jobs.tsv"
