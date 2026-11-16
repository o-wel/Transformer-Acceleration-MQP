#!/usr/bin/env bash
#SBATCH --account=isp
#SBATCH --partition=academic
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=mqp-fp16-report
set -eo pipefail
source "$HOME/MQP/Torch/setup.sh"
export PYTHONPATH="$MQP_REPORT_DEPS"
export TMPDIR="$MQP_RUN/report-tmp"
export MPLCONFIGDIR="$MQP_RUN/report-mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
python - "$MQP_REPORT_SOURCE" <<'PYCODE'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
for name,digest in json.loads((root/'SHA256.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
PYCODE
python "$MQP_REPORT_SOURCE/analyze.py" --out "$MQP_RUN"
python "$MQP_REPORT_SOURCE/build_report.py" --out "$MQP_RUN"
python "$MQP_REPORT_SOURCE/compare.py" "$MQP_SNAPSHOT"
