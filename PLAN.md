# Plan: Top-Tier LLM Vulnerability-Detection Papers → Reproduce → Ensemble → Cost-Aware Router

## Context

Prior work in this repo reproduced **QASecClaw** (arXiv:2605.01885) on OWASP Benchmark v1.2 —
Semgrep baseline matched near-exactly, LLM false-positive filter reproduced directionally
(51.3% FP reduction vs. the paper's 88.6%). A **ZeroFalse** (arXiv:2510.02534) run is
partially started on the GPU box. **Neither is a top-tier conference paper** — both are arXiv
preprints. This task moves up to venue-verified top-tier work and extends it in three stages:

1. **Reproduce** a top-tier LLM vulnerability-detection paper, substituting open-weight models
   where the paper used closed APIs.
2. **Replace the single LLM with an LLM collection** — no single model detects every
   vulnerability class well (already evidenced locally: qwen3.6:27b and qwen3-coder:30b, near
   identical size, differed by **47 percentage points** in suppression accuracy).
3. **Build a router** that dispatches each vulnerability query to the cheapest model competent
   for it, minimizing cost at slight accuracy/precision/F1 cost.

**Deliverable: publishable paper.** The gap being targeted is real — cost-aware cascading
(FrugalGPT, RouteLLM, Cluster-Route-Escalate) has never been applied to LLM vulnerability
detection, and no existing paper builds a CWE-conditioned model router.

**Decided:** ML venues (ICLR/NeurIPS) count as top-tier. **Not yet decided:** primary paper,
cost metric (options recorded in §5).

---

## 1. Candidate Papers — Venue-Verified Shortlist

### Tier A — best reproduction targets (public artifact + already report open-weight baselines)

| Paper | Venue | Artifact | Benchmark | LLMs in paper | Why it fits |
|---|---|---|---|---|---|
| **IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities** | **ICLR 2025** ✓confirmed | [iris-sast/iris](https://github.com/iris-sast/iris) + [CWE-Bench-Java](https://github.com/iris-sast/cwe-bench-java) | CWE-Bench-Java, 120 real CVEs (repo now 213 CVEs/49 CWEs) | GPT-4, GPT-3.5, **Llama-3 8B/70B, DeepSeekCoder-7B, Qwen2.5-Coder-32B, Gemma-2-27B** | **Strongest candidate.** Paper publishes exact open-weight numbers → no guessing. Two distinct LLM call sites (taint-spec inference; path FP filtering) = two routing surfaces. Four CWE classes with divergent per-model behavior = the ensemble thesis in one table. |
| **LLMDFA: Analyzing Dataflow in Code with LLMs** | **NeurIPS 2024** ✓confirmed | [chengpeng-wang/LLMDFA](https://github.com/chengpeng-wang/LLMDFA) | Obfuscated Juliet (DBZ, XSS, OSCI) | GPT-4-turbo, GPT-3.5-turbo, GPT-4o-mini | **Best-instrumented for the router phase.** Compilation-free (no Java builds, no CodeQL DBs — matters at 96% disk). Already emits **token cost + wall time per run**. Decomposes into 4 subtasks (source/sink extraction → intra-proc dataflow → SMT path feasibility → reporting), each independently routable. OpenAI-only in repo but it's a thin API layer → repoint at Ollama's OpenAI-compatible endpoint. |

**IRIS reported results to reproduce (120 CVEs):**

```
Model                #Detected  Rate     Avg FDR   Avg F1
CodeQL (baseline)       27      22.50%    90.03%   0.076
GPT-4                   55      45.83%    84.82%   0.177
GPT-3.5                 47      39.17%    90.42%   0.096
Llama-3 70B             54      45.00%    90.96%   0.113
DeepSeekCoder 7B        52      43.33%    95.40%   0.062
Qwen-2.5-Coder 32B      47      39.17%    92.38%   0.097
Gemma-2 27B             45      37.50%    91.23%   0.100
Llama-3 8B              41      34.17%    95.55%   0.058
```
CWE-Bench-Java composition: CWE-22 Path Traversal (55), CWE-79 XSS (31), CWE-94 Code
Injection (21), CWE-78 OS Command Injection (13).

### Tier B — strong top-tier, viable alternatives

| Paper | Venue | Artifact | Benchmark | LLMs | Notes |
|---|---|---|---|---|---|
| **LLMxCPG: Context-Aware Vulnerability Detection Through CPG-Guided LLMs** | **USENIX Security 2025** ✓confirmed (pp. 489–507) | [QCRI/LLMxCPG-D](https://huggingface.co/QCRI/LLMxCPG-D) (HF weights) | Traditional + verified C/C++ datasets | **QwQ-32B-Preview, fine-tuned** | Strictly-security top-tier. CPG slicing cuts code 67.8–90.9%; +15–40% F1 over SOTA. Downside for us: **fine-tuned**, so "swap the LLM" means retraining, not prompting — much more expensive to ensemble. |
| **Vulnerability Detection with Code Language Models: How Far Are We? (PrimeVul)** | **ICSE 2025** ✓confirmed | [PrimeVul dataset](https://github.com/DLVulDet/PrimeVul) | PrimeVul: 6,968 vuln + 228,800 benign fns, 140 CWEs | GPT-3.5, GPT-4, 7B code LMs | Cheapest to run (function-level, no builds). **But LLM results are near-random** (7B: 68.26% F1 on BigVul → 3.09% on PrimeVul) — almost no headroom for a router to trade against. Best used as a **hard realistic secondary testbed**, not the primary. |
| **GPTScan: Detecting Logic Vulnerabilities in Smart Contracts by Combining GPT with Program Analysis** | **ICSE 2024** ✓confirmed | [MetaTrustLabs/GPTScan](https://github.com/MetaTrustLabs/GPTScan) | ~400 contract projects, 3K Solidity files (Web3Bugs, DefiHacks) | GPT-3.5-turbo | Already **cost-motivated** ("20× cheaper than GPT-4", $0.01/KLoC, 14.39s/KLoC) — natural fit for a cost-routing extension. Different domain (Solidity), so no reuse of existing Java tooling. |
| **SV-TrustEval-C: Evaluating Structure and Semantic Reasoning in LLMs for Source Code Vulnerability Analysis** | **IEEE S&P 2025** ⚠️verify | [SV-TrustEval-C](https://github.com/Jackline97/SV-TrustEval-C) | C/C++ structure+semantics reasoning | multiple | Diagnostic benchmark — useful as **router feature engineering input** (tells you *what kind* of reasoning each model can do) rather than a reproduction target. |

### Tier C — supporting evidence, not reproduction targets

| Paper | Venue | Use to us |
|---|---|---|
| Understanding the Effectiveness of LLMs in Detecting Security Vulnerabilities (SecLLMHolmes) | ICST 2025 ⚠️(often miscited as S&P'24) — [seal-research/secvul-llm-study](https://github.com/seal-research/secvul-llm-study) | **16 LLMs × 5 datasets × 25 CWE classes.** Directly supplies the per-model/per-CWE complementarity matrix that motivates Phase B. Cite as motivation; don't reproduce. |
| Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection (JitVul) | ACL 2025 ⚠️verify | Repo-level benchmark, public. Possible generalization testbed. |
| From Large to Mammoth: Comparative Evaluation of LLMs in Vulnerability Detection | NDSS 2025 ⚠️verify | Cross-model comparison evidence for Phase B. |
| Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG | **TOSEM 2025** (journal, not FSE) | Knowledge-RAG baseline; a 2026 open-weight replication study already exists (arXiv:2606.04739). |
| FrugalGPT / RouteLLM (ICLR 2025) / Cluster-Route-Escalate (arXiv:2606.27457) | — | **Method ancestry for Phase C.** RouteLLM: 85% cost cut at 95% of GPT-4 quality on MT-Bench. None applied to vulnerability detection → our novelty claim. |

### Recommendation

**Primary: IRIS (ICLR 2025). Secondary: LLMDFA (NeurIPS 2024).**
IRIS gives exact open-weight reproduction targets and a repo-level realism story; LLMDFA gives
cheap iteration, built-in cost accounting, and a second independent domain to show the router
generalizes. Confirm before any work starts.

---

## 2. Phase A — Reproduce

- Clone IRIS + CWE-Bench-Java; pin to the **v1 branch** (the 120-CVE ICLR configuration), not
  the expanded 213-CVE main branch, so numbers are comparable to the published table.
- Build the Java projects and CodeQL DBs (`fetch_and_build.py` → `build_codeql_dbs.py` →
  `iris.py`). CodeQL 2.23.2. A `codeql-home/` already exists on the GPU box.
- Establish the **CodeQL-only baseline** first (target: 27/120, FDR 90.03%) — this is the
  cheap, LLM-free anchor, exactly as the Semgrep baseline anchored the QASecClaw work.
- Reproduce **each open-weight row** the paper reports, substituting the closest locally
  available Ollama model where the exact one is absent (see §6). Report substitutions
  explicitly in a table — the QASecClaw `FINDINGS.md` convention of disclosing every deviation
  (e.g. the CWE-326→327 synonym map) is the right standard and should carry over.
- Reuse the existing scoring approach in
  [score_qasecclaw.py](gpu-backup/QASecClaw/score_qasecclaw.py) /
  [score_semgrep.py](gpu-backup/QASecClaw/score_semgrep.py) — per-CWE precision/recall/F1/FDR
  breakdown tables, which IRIS also needs.
- **Carry forward the Ollama structured-output bug already discovered:** `think: true` + a
  constrained JSON `format` schema puts the answer in the `thinking` field and returns an empty
  `response`, causing 100% silent fail-opens. The fix (send `think: false`, plus a fallback
  that reads `thinking` when `response` is empty) is in
  [sast_filter_agent.py:129-148](gpu-backup/QASecClaw/sast_filter_agent.py#L129-L148) and must
  be reused verbatim.

**Success bar:** each open-weight row within a stated tolerance of the paper's detection count
and FDR; CodeQL baseline near-exact. Where it misses, diagnose (registry/ruleset drift, prompt
divergence, model substitution) rather than folding the gap in silently.

## 3. Phase B — Single LLM → LLM Collection

- Instrument both IRIS LLM call sites to accept a **model set** instead of one model.
- Run every candidate model over the full benchmark and build the **competence matrix**:
  `(model × CWE class × call site) → precision / recall / F1 / cost`. This matrix is both the
  paper's central empirical artifact and the router's training signal.
- Aggregation strategies to evaluate: majority vote, union (recall-max), intersection
  (precision-max), weighted vote by per-CWE competence, and confidence-weighted vote.
- Report the **complementarity gap**: CVEs detected by the union of models but by no single
  model. This is the number that justifies the whole ensemble direction.

## 4. Phase C — The Router

**Objective:** minimize cost subject to a bounded drop in F1 (e.g. ≤2 F1 points vs. the full
ensemble or the best single model).

Router designs, in increasing sophistication — evaluate all, they are the paper's ablation ladder:
1. **Static per-CWE routing** — argmin cost s.t. competence ≥ threshold, read straight off the
   Phase B matrix. Zero inference overhead. Likely a surprisingly strong baseline; must be
   reported honestly even if it wins.
2. **Learned classifier router** — features: CWE class, CodeQL rule ID, sink API, slice length,
   path depth, code embedding. Predicts the cheapest adequate model. This is the RouteLLM
   analogue, retrained on domain-specific quality data rather than chat preference data.
3. **Cascade with escalation** — cheap model first; escalate on low confidence, self-reported
   uncertainty, or disagreement within a cheap sub-ensemble. The FrugalGPT analogue.

**Required comparison points** (all on the same cost-accuracy axes):
- Oracle router (upper bound — cheapest model that is actually correct per query)
- Random router (lower bound)
- Always-cheapest / always-strongest single model
- Full ensemble (accuracy ceiling, cost ceiling)

**Headline figure:** cost–F1 **Pareto frontier**, router curve dominating the single-model points.

**Rigor for publication:** multi-seed variance (the QASecClaw work was explicitly single-run —
fix that here), held-out CWE generalization (train router on 3 CWE classes, test on the 4th),
and a router-feature ablation.

---

## 5. Cost Metric — Options To Decide

Everything runs locally, so there is no real API bill. Three framings:

| Option | What's reported | Pro | Con |
|---|---|---|---|
| **A. Both (suggested)** | Measured tokens + GPU wall-clock on the A100, **and** a dollar-equivalent using published per-token prices for each model's closest hosted tier | Local numbers are honest and reproducible; the $ figure is what reviewers and industry react to | Must defend the price mapping |
| **B. Local compute only** | GPU-seconds, prompt/completion tokens, peak VRAM | Fully reproducible, no pricing assumptions that go stale | Weaker practical framing for a paper |
| **C. API dollar proxy only** | Map each open-weight model to a published price point | Cleanest headline ("N% cheaper") | Mapping is an assumption doing a lot of load-bearing work |

Whichever is chosen, **instrument tokens and wall-clock from day one** — it costs nothing to
record and cannot be recovered retroactively. LLMDFA already emits both natively.

---

## 6. Infrastructure — Current State

**GPU box** (`gpuuser0_a@172.16.136.120:8022`): NVIDIA A100-SXM4-80GB, Ubuntu 24.04, no sudo.
Reachable; the local rsync mirror has been failing since 2026-08-15 (network, not auth).

**⚠️ Disk: 2.4T/2.5T used, 113 GB free (96%).** `~/.ollama` alone is 326 GB. Cleanup is a
prerequisite — CWE-Bench-Java builds plus CodeQL databases for 120 Java repos (300K LoC average,
10 repos >1M LoC) will not fit otherwise.

**Models already pulled** — a genuine ~40× cost ladder, which is exactly what a router needs:

| Tier | Models present |
|---|---|
| ~1–4B | smollm2:1.7b, qwen3:1.7b, qwen3:4b, llama3.2:1b/3b, gemma3:4b, qwen2.5-coder:1.5b/3b, nuextract:3.8b, phi3:3.8b |
| ~7–14B | llama3:8b, llama3.1:8b, qwen3:8b, qwen3:14b, phi4:14b, phi4-reasoning:14b, phi3:14b, deepseek-r1:8b/14b, mistral:7b, falcon3:7b/10b, llama2:13b |
| ~20–32B | gpt-oss:20b, mistral-small:24b, mistral-small3.2:24b, gemma4:26b, gemma3:27b, qwen3.6:27b, qwen3:30b, qwen3-coder:30b |
| ~70B | llama3.1:70b, llama3.3:70b |

**Pulls needed to match IRIS's reported rows:** `deepseek-coder:6.7b` (~4 GB),
`qwen2.5-coder:32b` (~20 GB), `gemma2:27b` (~16 GB). Llama-3 70B is approximated by
llama3.1:70b / llama3.3:70b — **a substitution to disclose, not paper over**.

---

## 7. Risks

- **Disk exhaustion** blocks IRIS before it starts. Resolve first, or start with LLMDFA
  (compilation-free, no CodeQL DBs).
- **Building 120 real Java repos is the classic failure point** of repo-level benchmarks —
  expect a nontrivial fraction to fail to build. Report the achieved subset explicitly.
- **Static per-CWE routing may match the learned router.** That is a legitimate finding and
  strengthens the paper if reported plainly; it does not weaken it.
- **Model substitution confounds reproduction.** QASecClaw showed two same-size models differing
  47 points — attribute gaps to substitution vs. prompt divergence vs. capability, don't merge them.
- **Ollama structured-output silent failure** already cost one full run. Fail-open counters and
  a batch-level sanity assert are mandatory instrumentation, not optional.

---

## 8. Verification

- Phase A: CodeQL baseline reproduces 27/120 detections and 90.03% FDR within tolerance; each
  open-weight row lands near its published detection count, with every deviation itemized.
- Phase B: competence matrix is complete (no missing model×CWE cells); the union-vs-best-single
  complementarity gap is measured and non-trivial.
- Phase C: router Pareto curve sits strictly between the random-router and oracle-router bounds;
  cost savings and F1 delta reported against all four comparison points; held-out-CWE result
  reported alongside in-distribution.
- Reproducibility: all metrics regenerable from saved JSON artifacts without a GPU, matching the
  existing pattern documented in [FINDINGS.md](gpu-backup/QASecClaw/FINDINGS.md#L154-L161).

---

## Next Step

**Nothing is to be executed yet.** The open decisions are: (a) confirm IRIS as primary and
LLMDFA as secondary, (b) pick the cost metric from §5, (c) approve GPU-box disk cleanup.
