# Probe runs on csecluster

`probe.py` over the pool is the project's critical path: nothing downstream
runs until the measurement table exists (README §Status).

## Why csecluster and not the standalone GPU boxes

`seconds` is the primary cost axis (problem doc §1.4) and the x-axis of every
frontier. `gpu0`/`gpu7` are containers on one shared host, and their GPUs carry
other users' work, so a probe there measures the neighbour's schedule as much as
the model's -- and because a sweep runs model by model over hours, that
contamination lands *unevenly across models*, which is a confound, not noise.
SLURM's `--gres` gives an exclusive device. All timed runs go here.

`gpu0` is still the right box for anything untimed: prompt iteration, parse-rate
shakedown, fitting router heads, training the S11 GNN.

## Layout on the cluster

    ~/routing/{src,cluster}       this repo, deployed by deploy.sh
    ~/data/primevul/*.jsonl       the six PrimeVul split files
    ~/data/units/*.jsonl          unit sets built by sample_units.py
    ~/job_results/<jobid>/        probe output + vLLM log per job
    ~/containers/vllm.sif         the serving image (shared with LLMDFA)
    ~/models/hf                   HF_HOME, weights pre-staged on the LOGIN node

## Run

    ./deploy.sh
    ssh csecluster 'python3 ~/routing/cluster/sample_units.py --split valid --pairs 100 --name pilot'
    ssh csecluster 'cd ~/cluster && MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct SHORT=qwen-1.5b \
        UNITS=~/data/units/pilot.jsonl sbatch ~/routing/cluster/run_probe.sbatch'

## Constraints that shape everything here

* **2 submitted jobs per user** (QOSMaxSubmitJobPerUserLimit). Queue in pairs.
* **A100-PCIE-40GB, 2 per node.** Anything over ~35 GB needs `TP=2 --gres=gpu:2`.
* **Home quota 50 GB soft / 200 GB hard**, ~106 GB already used by weights and
  the container. Stage one model at a time; a 32B is ~62 GB and will not fit
  alongside the current cache.
* **Compute nodes have no clean HTTPS** (TLS-intercepting proxy). Weights and
  data are staged on the login node; jobs run with `HF_HUB_OFFLINE=1`.
