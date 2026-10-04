#!/usr/bin/env python3
"""Patch the gpu7 IRIS copy for the scaled run. Every change is off unless its env var is set,
except the cache fix, which only makes concurrent writers safe (same content, atomic replace).
  IRIS_CODEQL_THREADS / IRIS_CODEQL_RAM   --threads / --ram for every CodeQL evaluation
  IRIS_CSV_NO_RERUN=1                     the CSV export reuses the SARIF run's cached result
                                          instead of evaluating the same query a second time
  IRIS_STOP_AFTER_LABELS=1                stop after the two LLM stages (3-4); a second run
                                          without it finds the labels on disk and does 5-9
Usage: patch_iris.py <IRIS root>   (idempotent)
"""
import os, sys

root = sys.argv[1]
MARK = "# [gpu7-scaled patch]"


def patch(path, pairs):
    s = open(path).read()
    if MARK in s:
        print("already patched:", path)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched:", path)


CQ = (f"{MARK}\n"
      "_CQ_OPTS = ([f\"--threads={os.environ['IRIS_CODEQL_THREADS']}\"] if os.getenv('IRIS_CODEQL_THREADS') else []) + \\\n"
      "           ([f\"--ram={os.environ['IRIS_CODEQL_RAM']}\"] if os.getenv('IRIS_CODEQL_RAM') else [])\n")

nv = os.path.join(root, "src/neusym_vul.py")
patch(nv, [
    ('CODEQL = f"{CODEQL_DIR}/codeql"\n',
     'CODEQL = f"{CODEQL_DIR}/codeql"\n' + CQ +
     "import fcntl, tempfile\n\n\n"
     "def _atomic_json_dump(obj, path):\n"
     "    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')\n"
     "    with os.fdopen(fd, 'w') as f:\n"
     "        json.dump(obj, f, indent=2)\n"
     "    os.replace(tmp, path)\n"),
    ("    def cache_llm_results(self, candidates, new_llm_result):\n"
     "        if os.path.exists(self.api_labels_cache_path):\n",
     "    def cache_llm_results(self, candidates, new_llm_result):\n"
     "        # projects of one CWE share this file and run concurrently: lock, then replace atomically\n"
     "        with open(self.api_labels_cache_path + '.lock', 'w') as _lk:\n"
     "            fcntl.flock(_lk, fcntl.LOCK_EX)\n"
     "            self._cache_llm_results_locked(candidates, new_llm_result)\n\n"
     "    def _cache_llm_results_locked(self, candidates, new_llm_result):\n"
     "        if os.path.exists(self.api_labels_cache_path):\n"),
    ('        json.dump(reload_cache, open(self.api_labels_cache_path, "w"), indent=2)\n',
     '        _atomic_json_dump(reload_cache, self.api_labels_cache_path)\n'),
    ('        sp.run([CODEQL, "database", "analyze", "--rerun", self.project_codeql_db_path, "--format=sarif-latest",',
     '        sp.run([CODEQL, "database", "analyze", "--rerun", *_CQ_OPTS, self.project_codeql_db_path, "--format=sarif-latest",'),
    ('        sp.run([CODEQL, "database", "analyze", "--rerun", self.project_codeql_db_path, "--format=csv",',
     '        sp.run([CODEQL, "database", "analyze", *([] if os.getenv("IRIS_CSV_NO_RERUN") == "1" else ["--rerun"]), *_CQ_OPTS, self.project_codeql_db_path, "--format=csv",'),
    ("        # 4. Query GPT for sources among internal function parameters\n"
     "        self.query_gpt_for_func_param_src()\n",
     "        # 4. Query GPT for sources among internal function parameters\n"
     "        self.query_gpt_for_func_param_src()\n"
     "        if os.getenv('IRIS_STOP_AFTER_LABELS') == '1':   # LLM stages only; a rerun skips 3-4\n"
     "            self.project_logger.info('==> IRIS_STOP_AFTER_LABELS: stopping after stage 4')\n"
     "            return\n"),
])

qr = os.path.join(root, "src/modules/codeql_query_runner.py")
patch(qr, [
    ('CODEQL = f"{CODEQL_DIR}/codeql"\n', 'CODEQL = f"{CODEQL_DIR}/codeql"\nimport os\n' + CQ),
    ('sp.run([CODEQL, "query", "run", f"--database={self.project_codeql_db_path}"',
     'sp.run([CODEQL, "query", "run", *_CQ_OPTS, f"--database={self.project_codeql_db_path}"'),
])
