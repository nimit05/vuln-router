#!/bin/bash
# Push the router code to csecluster. Data and weights are staged separately
# (login node only -- compute nodes cannot reach HuggingFace).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ssh csecluster 'mkdir -p ~/routing/cluster ~/routing/src ~/data/units'
scp -q -r "$HERE/src/vulnrouter" csecluster:~/routing/src/
scp -q "$HERE/cluster/probe_client.py" "$HERE/cluster/sample_units.py" \
       "$HERE/cluster/run_probe.sbatch" csecluster:~/routing/cluster/
echo "deployed to csecluster:~/routing"
