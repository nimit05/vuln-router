#!/bin/bash
# Pulls the GPU server's working tree into ./gpu-backup every 5 minutes.
# Additive mirror (no --delete): files removed on the server are retained locally.
REMOTE="gpuuser0_a@172.16.136.120:~/nimit/vuln-pred-results/"
LOCAL="/Users/nimitwadhwa/Documents/projects/vuln-pred-results/gpu-backup/"
LOG="/Users/nimitwadhwa/Documents/projects/vuln-pred-results/gpu-backup-sync.log"

# Skip re-creatable bulk: virtualenvs, caches, CodeQL databases, Maven/Java build output.
EXCLUDES=(
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude 'codeql-home/'
  --exclude '*codeql-db*/'
  --exclude '*.codeqldb/'
  --exclude 'target/'
  --exclude '*.jar'
  --exclude '*.tar.gz'
  --exclude '*.zip'
  --exclude '.cache/'
  --exclude 'ZeroFalse/'
  # QASecClaw finished; its artifacts are preserved in results/QASecClaw/ and the
  # 239M owasp-benchmark checkout is re-clonable. Stop re-pulling it.
  --exclude 'QASecClaw/'
  # IRIS reproduction: re-creatable bulk. Keep source, scripts, output/ and results/.
  --exclude 'IRIS/codeql/'            # patched CodeQL CLI install, ~2.1G
  --exclude 'IRIS/.conda-iris/'       # conda env
  --exclude 'IRIS/data/codeql-dbs/'   # generated CodeQL databases
  --exclude 'project-sources/'        # 120 cloned Java repos, ~6.6G
  --exclude 'java-env/'               # JDK/Maven/Gradle distributions
)

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  if rsync -az "${EXCLUDES[@]}" -e "ssh -p 8022" "$REMOTE" "$LOCAL" >> "$LOG" 2>&1; then
    echo "$ts sync OK" >> "$LOG"
  else
    echo "$ts sync FAILED" >> "$LOG"
  fi
  sleep 300
done
