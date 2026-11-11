#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python -m pip install --no-deps --no-compile --target "$study_dir/.plot_deps" -r "$study_dir/plot-requirements.txt"
# Keep the GPU environment's existing NumPy and packaging packages.
for name in PIL contourpy cycler dateutil fontTools kiwisolver matplotlib mpl_toolkits pillow.libs pylab.py pyparsing tzdata; do
    ln -sfn ".plot_deps/$name" "$study_dir/$name"
done
