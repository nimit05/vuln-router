#!/bin/bash
# Applies the reproduction patches to an LLMDFA checkout. Idempotent.
# Usage: ./patch_llmdfa.sh /path/to/LLMDFA
set -euo pipefail

REPO="${1:?usage: patch_llmdfa.sh /path/to/LLMDFA}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO/src"

[ -f "$SRC/run_llmdfa.py" ] || { echo "not an LLMDFA checkout: $REPO" >&2; exit 1; }

# --- 1. local inference backend for the main pipeline ----------------------
[ -f "$SRC/utility/llm.py.orig" ] || cp "$SRC/utility/llm.py" "$SRC/utility/llm.py.orig"
cp "$HERE/llm_local.py" "$SRC/utility/llm.py"
echo "patched: utility/llm.py (original kept as llm.py.orig)"

# --- 2. bound the unbounded LLM retry loops --------------------------------
# Both loops `continue` forever when the response contains no bare Yes/No.
# At temperature 0 that is an infinite, billing, non-terminating loop.
# On exhaustion the surrounding code falls through fail-closed (no bug reported).
patch_loop () {
  local f="$1" indent="$2"
  grep -q "^${indent}while True:$" "$f" || { echo "  $(basename "$f"): loop already bounded"; return 0; }
  python3 - "$f" "$indent" <<'PYEOF'
import sys, io
path, indent = sys.argv[1], sys.argv[2]
s = io.open(path, encoding="utf-8").read()
if not s.startswith("import os"):
    s = "import os\n" + s
s = s.replace(
    "\n" + indent + "while True:\n",
    "\n" + indent + 'for _attempt in range(int(os.environ.get("LLMDFA_MAX_RETRY", "5"))):\n',
    1,
)
io.open(path, "w", encoding="utf-8").write(s)
PYEOF
  echo "patched: $(basename "$f") retry loop"
}

patch_loop "$SRC/LMAgent/flow/intra_flow_propagator.py" "                "
patch_loop "$SRC/LMAgent/flow/inter_flow_validator.py"  "        "

# --- 3. the extractor synthesizer's SEPARATE llm layer ---------------------
# TSAgent/synthesis/ does not use utility/llm.py at all. It has its own GPT class
# with a hardcoded OpenAI client and tiktoken.encoding_for_model(), which raises
# on any non-OpenAI model name, plus a select_model() that rejects unknown names.
# Phase I cannot run on a local model without this.
python3 - "$SRC" <<'PYEOF'
import io, sys
src = sys.argv[1]

p = src + "/TSAgent/synthesis/llm.py"
s = io.open(p, encoding="utf-8").read()
if "LLMDFA_BASE_URL" not in s:
    if not s.startswith("import os"):
        s = "import os\n" + s
    old = (
        "        self.client = openai.OpenAI(\n"
        "            api_key=self.api_key,\n"
        "        )\n"
        "        self.encoder = tiktoken.encoding_for_model(self.model)"
    )
    new = (
        "        self.client = openai.OpenAI(\n"
        "            api_key=self.api_key or \"EMPTY\",\n"
        "            base_url=os.environ.get(\"LLMDFA_BASE_URL\", \"https://api.openai.com/v1\"),\n"
        "        )\n"
        "        try:\n"
        "            self.encoder = tiktoken.encoding_for_model(self.model)\n"
        "        except Exception:\n"
        "            self.encoder = tiktoken.get_encoding(\"cl100k_base\")"
    )
    assert old in s, "synthesis/llm.py: GPT.__init__ shape changed"
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: TSAgent/synthesis/llm.py")
else:
    print("  synthesis/llm.py already patched")

p = src + "/TSAgent/synthesis/main.py"
s = io.open(p, encoding="utf-8").read()
if "fall through to a local endpoint" not in s:
    old = "    else:\n        raise ValueError(f\"Unknown model name {name}\")"
    new = (
        "    else:\n"
        "        # unknown names fall through to a local endpoint (vLLM) rather than raising\n"
        "        model = GPT(\n"
        "            api_key=os.environ.get(\"OPENAI_API_KEY\", \"EMPTY\"),\n"
        "            model=name,\n"
        "            temperature=temperature,\n"
        "            log_file=output_file,\n"
        "        )"
    )
    assert old in s, "synthesis/main.py: select_model shape changed"
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: TSAgent/synthesis/main.py")
else:
    print("  synthesis/main.py already patched")
PYEOF

echo
echo "Grammar .so: build once with  \$VENV/bin/python lib/build.py  (needs gcc)"

# --- 4. argparse whitelist on --model --------------------------------------
# select_model() now accepts any name, but argparse rejects it first.
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/TSAgent/synthesis/main.py"
s = io.open(p, encoding="utf-8").read()
old = '        choices=["gpt3.5", "gpt4", "gpt-4o-mini", "gemini-pro", "claude-haiku"],\n'
if old in s:
    s = s.replace(old, "", 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: synthesis/main.py --model choices removed")
else:
    print("  --model choices already removed")
PYEOF

# --- 5. optional provider imports in the synthesizer ------------------------
# synthesis/llm.py imports google.generativeai and anthropic at module level.
# We serve locally and do not install them, so the import kills Phase I.
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/TSAgent/synthesis/llm.py"
s = io.open(p, encoding="utf-8").read()
if "_HAS_GENAI" not in s:
    old = "import google.generativeai as genai\nimport anthropic\n"
    new = ("try:\n"
           "    import google.generativeai as genai\n"
           "    _HAS_GENAI = True\n"
           "except Exception:\n"
           "    genai = None\n"
           "    _HAS_GENAI = False\n"
           "try:\n"
           "    import anthropic\n"
           "    _HAS_ANTHROPIC = True\n"
           "except Exception:\n"
           "    anthropic = None\n"
           "    _HAS_ANTHROPIC = False\n")
    assert old in s, "synthesis/llm.py: import block shape changed"
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: synthesis/llm.py optional provider imports")
else:
    print("  synthesis/llm.py imports already optional")
PYEOF

# --- 6. argparse whitelist on run_llmdfa.py --model-name -------------------
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/run_llmdfa.py"
s = io.open(p, encoding="utf-8").read()
old = '        "--model-name",\n        choices=models,\n'
if old in s:
    s = s.replace(old, '        "--model-name",\n', 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: run_llmdfa.py --model-name choices removed")
else:
    print("  run_llmdfa.py --model-name already open")
PYEOF

# --- 7. configurable case limit + deterministic sampling -------------------
# Upstream offers only `single` (hard-coded first 10 cases) or `all` (everything).
# Gate C needs the paper's 37-program protocol, and the first N files in walk order
# are all from the same subdirectory (s01), which is a biased sample.
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/run_llmdfa.py"
s = io.open(p, encoding="utf-8").read()

old_break = '            if DFA_num > 10 and self.analysis_mode == "single":'
new_break = ('            if DFA_num > int(os.environ.get("LLMDFA_CASE_LIMIT", "10")) '
             'and self.analysis_mode == "single":')
if old_break in s:
    s = s.replace(old_break, new_break, 1)
    print("patched: run_llmdfa.py case limit -> LLMDFA_CASE_LIMIT")
else:
    print("  case limit already configurable")

old_loop = "        # for java_file in self.analyzed_java_files:\n        for java_file in self.all_single_files:"
new_loop = ("        _seed = os.environ.get(\"LLMDFA_CASE_SEED\")\n"
            "        if _seed:\n"
            "            import random\n"
            "            random.Random(int(_seed)).shuffle(self.all_single_files)\n"
            "\n"
            "        # for java_file in self.analyzed_java_files:\n"
            "        for java_file in self.all_single_files:")
if "LLMDFA_CASE_SEED" not in s:
    assert old_loop in s, "run_llmdfa.py: case loop shape changed"
    s = s.replace(old_loop, new_loop, 1)
    print("patched: run_llmdfa.py deterministic case sampling -> LLMDFA_CASE_SEED")
else:
    print("  case sampling already patched")

io.open(p, "w", encoding="utf-8").write(s)
PYEOF

# --- 8. case sharding for long runs ---------------------------------------
# DBZ is ~1850 cases at ~40s each = ~21h in one uninterruptible job. Cases are
# fully independent (each builds its own DFA engine), so they can be split across
# jobs and concatenated. Ordering is sorted ONLY when sharding is requested, so
# earlier CASE_SEED samples stay bit-for-bit reproducible.
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/run_llmdfa.py"
s = io.open(p, encoding="utf-8").read()
if "LLMDFA_CASE_OFFSET" not in s:
    anchor = '        _seed = os.environ.get("LLMDFA_CASE_SEED")'
    shard = ('        _off = int(os.environ.get("LLMDFA_CASE_OFFSET", "0"))\n'
             '        _cnt = os.environ.get("LLMDFA_CASE_COUNT")\n'
             '        if _off or _cnt:\n'
             '            self.all_single_files = sorted(self.all_single_files)\n'
             '            _end = _off + int(_cnt) if _cnt else None\n'
             '            self.all_single_files = self.all_single_files[_off:_end]\n'
             '            print("SHARD offset=%s count=%s -> %d cases"\n'
             '                  % (_off, _cnt, len(self.all_single_files)), flush=True)\n'
             '\n')
    assert anchor in s, "run_llmdfa.py: sampling block shape changed"
    s = s.replace(anchor, shard + anchor, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: run_llmdfa.py case sharding -> LLMDFA_CASE_OFFSET/COUNT")
else:
    print("  sharding already patched")
PYEOF

# --- 9. crash on unparseable Z3 response -----------------------------------
# construct_solving_program() ends with `assert program != ""`, so if the model fails
# three times to wrap its script in exactly two backticks, the ENTIRE run dies.
# The paper describes a fallback, not a crash: "If the script is buggy after three
# trials, LLMDFA enforces LLMs to determine path feasibility based on the path
# information." Returning "" reaches exactly that path (empty program -> no SAT/UNSAT
# -> is_uncertain -> None -> LLM-based check), so this makes the code match the paper.
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/LMAgent/flow/inter_flow_validator.py"
s = io.open(p, encoding="utf-8").read()
old = "        assert program != \"\"\n        return program"
new = ("        if program == \"\":\n"
       "            print(\"[LLMDFA] Z3 script unparseable after 3 tries; \"\n"
       "                  \"falling back to LLM path check (paper Sec 3.3)\", flush=True)\n"
       "        return program")
if old in s:
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: inter_flow_validator.py assert -> paper-described fallback")
else:
    print("  Z3 assert already handled")
PYEOF

# --- 10. bound the extractor-synthesis refinement loop ---------------------
# synthesize.py runs `while feedback is not None:` with no iteration cap. It exits
# only when the synthesized parser is PERFECT on the spec examples. A model that
# cannot reach perfection loops until SLURM walltime: Phi-4-mini-instruct reached
# 593 refinement rounds on xss_sink in 33 minutes with no sign of converging
# (job 2232), while Qwen2.5-Coder-7B needs 1 iteration and zero fixes.
#
# The paper's own worst observed case is claude-3 on the XSS sink extractor:
# NFP 0.00%, max 30 fixes, avg 7.20 (Table 6). A cap of 40 therefore sits above
# every convergence the paper reports, so it cannot truncate a run that would
# have succeeded. On exhaustion we raise rather than return a broken parser: a
# wrong extractor silently poisons every later phase, and "this model cannot do
# Phase I" is itself the result we want recorded.
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/TSAgent/synthesis/synthesize.py"
s = io.open(p, encoding="utf-8").read()
if "LLMDFA_SYN_MAX_ITER" not in s:
    if not s.startswith("import os"):
        s = "import os\n" + s
    old = "    iterations = 1\n    while feedback is not None:\n"
    new = ('    iterations = 1\n'
           '    _syn_cap = int(os.environ.get("LLMDFA_SYN_MAX_ITER", "40"))\n'
           '    while feedback is not None:\n'
           '        if iterations >= _syn_cap:\n'
           '            raise RuntimeError(\n'
           '                "[LLMDFA] extractor synthesis did not converge for %s after %d "\n'
           '                "iterations (cap LLMDFA_SYN_MAX_ITER=%d). The paper\'s worst case "\n'
           '                "is 30 fixes (Table 6); this model cannot synthesize this extractor."\n'
           '                % (getattr(spec, "name", "?"), iterations, _syn_cap))\n')
    assert old in s, "synthesize.py: refinement loop shape changed"
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: synthesize.py refinement loop -> LLMDFA_SYN_MAX_ITER (default 40)")
else:
    print("  synthesize.py loop already bounded")
PYEOF

# --- 11. make seeded sampling reproducible across nodes --------------------
# Section 7 shuffles self.all_single_files with a seeded Random, but that list
# comes straight from a filesystem walk, whose order is not guaranteed stable
# across nodes or filesystems. Same seed + different walk order = a different
# sample, which silently breaks any cross-model comparison drawn on a sample.
# Sorting first makes the seeded sample a pure function of (seed, benchmark).
# Section 8's OFFSET/COUNT path already sorts, so this only fixes the seed path.
python3 - "$SRC" <<'PYEOF'
import io, sys
p = sys.argv[1] + "/run_llmdfa.py"
s = io.open(p, encoding="utf-8").read()
old = ("        if _seed:\n"
       "            import random\n"
       "            random.Random(int(_seed)).shuffle(self.all_single_files)\n")
new = ("        if _seed:\n"
       "            import random\n"
       "            # sort first: walk order is not stable across nodes, so an\n"
       "            # unsorted seeded shuffle is not reproducible between runs\n"
       "            self.all_single_files = sorted(self.all_single_files)\n"
       "            random.Random(int(_seed)).shuffle(self.all_single_files)\n")
if "walk order is not stable" in s:
    print("  seeded sampling already sorted")
elif old in s:
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: run_llmdfa.py seeded sampling now sorts before shuffling")
else:
    raise SystemExit("run_llmdfa.py: seeded shuffle block shape changed")
PYEOF
