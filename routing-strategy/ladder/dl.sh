#!/bin/bash
# parallel curl downloads into /tmp/nimit/models (HF snapshot_download is slow on these boxes)
set -u
for repo in Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B openai/gpt-oss-120b; do
  d=/tmp/nimit/models/$(basename $repo); mkdir -p $d; cd $d
  echo "=== $repo $(date)"
  files=$(curl -s https://huggingface.co/api/models/$repo | python3 -c '
import json,sys
for s in json.load(sys.stdin)["siblings"]:
    f=s["rfilename"]
    if f.startswith(("original/","metal/")): continue
    if f.endswith((".json",".safetensors",".jinja",".txt",".model")) or f.startswith("tokenizer"): print(f)')
  for f in $files; do
    [ -s "$f" ] && continue
    (curl -sfL --retry 5 -C - -o "$f.part" "https://huggingface.co/$repo/resolve/main/$f" && mv "$f.part" "$f" && echo "  got $f $(date +%T)") &
  done
  wait
  echo "=== done $repo $(date)"; du -sh $d
done
echo "=== ALL DOWNLOADED $(date)"
