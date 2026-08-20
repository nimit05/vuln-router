# IRIS Reproduction — Environment & Deviations Log

Target: *IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities*, ICLR 2025
(arXiv:2405.17238). Repo pinned to branch `v1` (the 120-CVE ICLR configuration).

Run start: 2026-08-17
Hardware: NVIDIA A100-SXM4-80GB, 256 cores, 2TB RAM, Ubuntu 24.04, no sudo (user-space only).

## Environment as built

| Component | Paper | Ours | Note |
|---|---|---|---|
| CodeQL | 2.15.3 | **2.15.5 (IRIS patched release `codeql-0.8.3-patched`)** | The repo's own setup script installs 2.15.5; 807,846,025 bytes, size-verified. IRIS states this patched build **is required**. |
| JDK 8u202 | Oracle | **Azul Zulu 8.36.0.1-CA (1.8.0_202-b05)** | Oracle 8u202 is auth-gated (redirects to Oracle SSO). Zulu is the same OpenJDK update level. |
| JDK 17 | Oracle | **Oracle JDK 17 (17+35-LTS-2724)** | Downloads without an account; exact file the scripts expect. |
| JDK 7u80 | Oracle | Azul Zulu 7.0.352 | **Used by zero projects** (build_info: 8u202 x92, 17 x28). Present only to stop `setup_jdk.py` aborting. |
| Maven | 3.2.1 / 3.5.0 / 3.9.8 | same | 3.9.8 re-sourced from Maven Central (see bug 2). |
| Gradle | 6.8.2 / 7.6.4 / 8.9 | same | |
| LLM serving | HuggingFace transformers, full precision | **Ollama, Q4_0 quantized** | Biggest deviation. See "Quantization" below. |

## Upstream bugs hit and fixed

1. **`setup_jdk.py` does not skip a missing JDK tarball.** It prints "NOT found; skipping"
   but then untars anyway, and `exit(1)` on tar failure aborts the entire `setup.py` run.
   Worked around by supplying all three JDK directories with the exact expected names.

2. **Maven 3.9.8 URL is dead.** `mvn_version.json` points at
   `archive.apache.org/dist/maven/maven-3/3.9.8/...`, which did not serve the file;
   `setup_mvn.py` then crashed with `FileNotFoundError: 'mvn'` and aborted the whole build.
   Re-sourced from `repo.maven.apache.org`.

3. **`build_codeql_dbs.py` never puts Gradle on PATH, and adds Maven only when
   `build_info.csv` records a maven version.** CodeQL's autobuild sniffs the source tree
   independently: a project recorded as Gradle but shipping a `pom.xml` (zip4j) makes
   autobuild call `mvn`, producing `Cannot run program "mvn"`. Patched to append all
   installed Maven/Gradle bins as fallbacks *after* any recorded version.

4. **`data/build_info.csv` disagrees with what actually builds locally.** `build_one.py`
   walks a 10-attempt toolchain ladder and dumps the winner to `build-info/<slug>.json`.
   Patched `build_codeql_dbs.py` to prefer that record, and to pass CodeQL an explicit
   `--command` for gradle-built projects instead of letting autobuild guess.

## Quantization — the material deviation

The paper ran "instruction-tuned versions of four open-source LLMs via huggingface API",
i.e. full-precision weights. Full-precision serving was not feasible here: gemma-2-27b
(~54GB) + Qwen2.5-Coder-32B (~64GB) + Llama-3-70B (~140GB) exceed available disk (~48GB
free on a shared 2.5TB volume that is 99% full and mostly other users' data), and Llama-3
and Gemma-2 are additionally license-gated on HuggingFace.

Ollama Q4_0 was used instead. This is **not cosmetic**: at Q4_0, `deepseek-coder:6.7b`
ignored IRIS's "DO NOT OUTPUT ANYTHING OTHER THAN JSON" instruction and returned English
prose. IRIS's parser (`re.findall(r"\[[\s\S]*\]", s)[0]`) silently dropped nearly
everything, yielding 4 sources / 4 sinks and **zero** detected paths.

Note: `deepseek-coder:6.7b` and `deepseek-coder:6.7b-instruct` share digest `ce298d984115`
— they are the same instruction-tuned model. The failure is precision, not base-vs-instruct.

### Fix: schema-constrained decoding (added, disclose in any write-up)
`src/models/ollama.py` now selects an Ollama structured-output JSON schema per stage,
keyed off the system prompt:
- API labelling  -> array of {package, class, method, signature, sink_args, type}
- Func-param labelling -> array of {package, class, method, signature, tainted_input}
- Posthoc contextual filtering -> unconstrained (IRIS's original behaviour)

Effect on `perwendel__spark_CVE-2018-9159_2.7.1` / CWE-022 / deepseek-coder 6.7b:

| | sources | sinks | taint-propagators | func-param sources |
|---|---|---|---|---|
| Q4_0, unconstrained (IRIS as-shipped) | 4 | 4 | 6 | 8 |
| Q4_0, generic `format="json"` | 0 | 0 | 0 | - |
| **Q4_0, per-stage schema** | **17** | **17** | **32** | **47** |

Generic JSON mode is *worse* than no constraint: models emit a valid JSON **object**, but
IRIS's regex requires a top-level **array**, so everything is discarded.

Also set `temperature: 0.0` (IRIS's Ollama backend defaults to 0.8) for determinism.

## Maven Central rate limiting (HTTP 429) — affects build AND database yield

CodeQL's Java autobuild re-runs the project build (`mvn clean package ...`) while tracing.
That means the network dependency resolution happens *twice*: once for `build_one.py`, once
for `codeql database create`.

Running `setup.py` with its default `ThreadPoolExecutor()` fan-out (min(32, cpu+4) = **32
concurrent Maven builds** on this 256-core box) triggered sustained throttling from
`repo.maven.apache.org`:

```
[ERROR] Failed to execute goal ...maven-dependency-plugin:2.8:copy (default-cli):
  Could not transfer artifact org.apache.maven:apache-maven:zip:bin:3.8.1 ...
  Waited too long to access: ... Return code is: 429
ERROR: Failed to download Maven version 3.8.1
```

Observed counts: **58** rate-limit events during the parallel build phase, and further
429s during (serial) database creation.

Consequences for anyone reproducing:
- Database-creation failures are **not** evidence that a project is unbuildable. They are
  frequently transient network throttling and succeed on a later retry.
- Do not report a detection denominator based on a single pass. Retry failures before
  fixing the achieved subset.
- Prefer low build concurrency. The 32-way default is counter-productive here: it converts
  a bandwidth-bound job into a throttled one.
- `~/.m2` reached 9.1G, so most artifacts are cached locally; a retry pass resolves far
  more from cache and hits the network less.

Mitigation applied: databases are built **serially** by a driver script
(`/tmp/build_dbs.sh`) that records per-project outcome, database size and wall time to
`results/IRIS/db_build_status.csv`, enforces a 12GB free-disk floor, and deletes partial
databases so failures are cleanly retryable.
