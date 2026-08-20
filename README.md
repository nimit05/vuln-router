# vuln-router

IRIS reproduction — the primary reproduction target for cost-aware LLM routing
in vulnerability detection.

Upstream: [iris-sast/iris](https://github.com/iris-sast/iris) @ `v1`
(the 120-CVE ICLR 2025 configuration), MIT licensed, (c) 2024 Ziyang Li.

## Layout

| Path | Contents |
|---|---|
| `IRIS/src/` | pipeline — `neusym_vul.py`, `codeql_vul.py`, `prompts.py`, `modules/`, 16 model adapters |
| `IRIS/scripts/` | `build_codeql_dbs.py`, `get_packages_codeql.py` |
| `IRIS/results/` | upstream's published per-model CSVs |
| `IRIS/reproduction/` | this reproduction's findings, deviations, checkpoints, metrics, run logs |

## Modified vs upstream

    scripts/build_codeql_dbs.py
    src/codeql_vul.py
    src/neusym_vul.py
    src/models/{config,deepseek,llm,ollama}.py

## Not in this repo

Excluded via `.gitignore` as re-creatable (~60 GB): `IRIS/data/` (CWE-Bench-Java
sources and the JDK/Maven/Gradle toolchain), `IRIS/.conda-iris/`, `IRIS/codeql/`
(CodeQL CLI install), `IRIS/output/` (per-CVE run output).

`IRIS/data/cwe-bench-java` is a submodule of
[iris-sast/cwe-bench-java](https://github.com/iris-sast/cwe-bench-java) — init it
separately after cloning.

## Reproduction status

See `IRIS/reproduction/FINDINGS.md`, `RESULTS_TABLE.md`, and `SETUP_DEVIATIONS.md`.
