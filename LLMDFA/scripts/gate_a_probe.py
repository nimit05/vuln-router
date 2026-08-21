"""Gate A: protocol-compliance probe for LLMDFA.

Does NOT touch the benchmark. Answers one question: can this model drive the
pipeline's three output protocols at all?

  1. Phase II  - does it emit a bare Yes/No that LLMDFA's parser can read?
  2. Phase IV  - does it emit a Z3 script that is fenced correctly AND runs?
  (Phase I, extractor synthesis, is exercised separately by running LLMDFA's
   own TSAgent/synthesis/main.py, which is real rather than simulated.)

Prompts are built from LLMDFA's own prompt configs and its own
construct_solving_program() text, so this measures the real formats.

Usage:
  LLMDFA_BASE_URL=http://127.0.0.1:8000/v1 python gate_a_probe.py \
      --llmdfa ~/LLMDFA --model <served-name> --out gate_a.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# --- Phase II probe items: tiny Java snippets with unambiguous ground truth ---
YESNO_ITEMS = [
    # (program, src_name, src_line, sink_name, sink_line, expected)
    ("""1. String a = req.getParameter("x");
2. String b = a;
3. resp.getWriter().write(b);""", "a", 1, "b", 3, "Yes"),
    ("""1. String a = req.getParameter("x");
2. String c = "constant";
3. resp.getWriter().write(c);""", "a", 1, "c", 3, "No"),
    ("""1. String a = req.getParameter("name");
2. String b = a.trim();
3. String c = b.toUpperCase();
4. resp.getWriter().write(c);""", "a", 1, "c", 4, "Yes"),
    ("""1. String a = req.getParameter("name");
2. String b = "hello";
3. String c = b + "world";
4. resp.getWriter().write(c);""", "a", 1, "c", 4, "No"),
    ("""1. String a = req.getParameter("q");
2. StringBuilder sb = new StringBuilder();
3. sb.append(a);
4. resp.getWriter().write(sb.toString());""", "a", 1, "sb", 4, "Yes"),
    ("""1. String a = req.getParameter("q");
2. int n = 5;
3. int m = n * 2;
4. resp.getWriter().write(String.valueOf(m));""", "a", 1, "m", 4, "No"),
]

# --- Phase IV probe items: path descriptions with a known SAT/UNSAT answer ----
Z3_ITEMS = [
    ("The variable b is 0. The line 4 is in the true branch of the if-statement "
     "whose condition is b > 1.", 4, "0", "UNSAT"),
    ("The variable b is 0. The line 4 is in the true branch of the if-statement "
     "whose condition is b < 1.", 4, "0", "SAT"),
    ("The variable data is 0. The line 7 is in the else branch of the if-statement "
     "whose condition is data != 0.", 7, "0", "SAT"),
]


def build_yesno_prompt(cfg, program, src, src_line, sink, sink_line):
    """Mirrors IntraFlowPropagator.apply()'s prompt construction exactly."""
    prompt = cfg["task"]
    prompt += "\n" + "\n".join(cfg["analysis_rules"])
    prompt += "\n" + "\n".join(cfg["analysis_examples"])
    prompt += "\n" + "".join(cfg["meta_prompts"])

    question = (
        cfg["question_template"]
        .replace("<SRC_NAME>", src)
        .replace("<SRC_LINE>", str(src_line))
        .replace("<SINK_NAME>", sink)
        .replace("<SINK_LINE>", str(sink_line))
        .replace("<CMP>", "the same" if src == sink else "different")
        .replace("<USED>", "used")
    )
    prompt = prompt.replace("<PROGRAM>", program)
    prompt = prompt.replace("<QUESTION>", question)
    prompt = prompt.replace("<ANSWER>", "\n".join(cfg["answer_format_cot"]))
    return cfg["system_role"], prompt


def build_z3_prompt(line_number, path_description, val_literal):
    """Mirrors InterFlowValidator.construct_solving_program()'s message."""
    skeleton = """
        from z3 import *

        def path_constraint_solving():
            solver = Solver()

            # TODO: Insert your implementation here

            # The end of your implementation

            result = solver.check()

            if result == sat:
                print("SAT")
            elif result == unsat:
                print("UNSAT")
            else:
                print("Result is unknown")

        # Call the function
        path_constraint_solving()
        """
    m = "Please write the path condition for the line %d based on the description: \n" % line_number
    m += path_description + "\n"
    m += ("Please write a runnable Python code to solve the path condition using Z3 python binding. "
          "Once I execute the program, the result can indicate whether the path constraint is SAT or UNSAT.\n")
    m += "To convenience your implementation, I provide the following skeleton for you:\n"
    m += skeleton + "\n"
    m += 'Please write your own code after the comment "TODO: Insert your implementation here". \n\n'
    m += ("- If the line is in the true branch of an if-statement with condition C, add solver.add(C).\n"
          "- If it is in the else branch, add solver.add(C == False).\n")
    if val_literal:
        m += "- Remember to append the constraint that the focused variable equals %s.\n\n" % val_literal
    m += ("DO NOT change other program structures in the skeleton. "
          "Your response should contain the COMPLETED WHOLE PROGRAM only and DO NOT add extra explanations. "
          "Also, the program should be wrapped with a pair of ``` at the beginning and the end of the program.\n")
    return "You are a Java programmer good at reasoning about path conditions.", m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmdfa", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="gate_a.json")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.expanduser(args.llmdfa), "src"))
    from utility.llm import LLM              # our patched, vLLM-backed client
    from LMAgent.LM_agent import LMAgent     # the real Yes/No parser

    cfg_path = os.path.join(os.path.expanduser(args.llmdfa),
                            "src/prompt/flow/dep_flow_propagator.json")
    cfg = json.load(open(cfg_path))

    report = {"model": args.model, "base_url": os.environ.get("LLMDFA_BASE_URL")}

    # ---- Phase II: Yes/No protocol ----
    print("=== Phase II probe: Yes/No compliance ===", flush=True)
    parsed = correct = 0
    lat, in_tok, out_tok = [], 0, 0
    raw_samples = []
    for i, (prog, s, sl, k, kl, expected) in enumerate(YESNO_ITEMS, 1):
        role, prompt = build_yesno_prompt(cfg, prog, s, sl, k, kl)
        m = LLM(args.model, "EMPTY", 0.0, role)
        t0 = time.time()
        out, it, ot = m.infer(prompt)
        lat.append(time.time() - t0)
        in_tok += it; out_tok += ot
        vec = LMAgent.process_yes_no_list_in_response(out)
        ok = len(vec) > 0
        if not ok:
            # keep the raw text: an unparseable answer is a FORMAT failure, which is
            # a different finding from a wrong answer, and the two must not be merged
            raw_samples.append(out[:400])
        parsed += ok
        hit = ok and vec[0] == expected
        correct += hit
        print("  item %d: parsed=%s answer=%s expected=%s (%.1fs)"
              % (i, ok, vec[0] if ok else "-", expected, lat[-1]), flush=True)

    report["phase2_yesno"] = {
        "n": len(YESNO_ITEMS),
        "parse_rate": parsed / len(YESNO_ITEMS),
        "accuracy": correct / len(YESNO_ITEMS),
        "avg_latency_s": sum(lat) / len(lat),
        "input_tokens": in_tok, "output_tokens": out_tok,
        "unparseable_samples": raw_samples,
    }

    # ---- Phase IV: Z3 script protocol ----
    print("=== Phase IV probe: Z3 script synthesis ===", flush=True)
    fenced = ran = right = 0
    zlat, zin, zout = [], 0, 0
    errors = []
    for i, (desc, line, lit, expected) in enumerate(Z3_ITEMS, 1):
        role, prompt = build_z3_prompt(line, desc, lit)
        m = LLM(args.model, "EMPTY", 0.0, role)
        t0 = time.time()
        out, it, ot = m.infer(prompt)
        zlat.append(time.time() - t0)
        zin += it; zout += ot
        resp = out.replace("```python", "```")
        ok_fence = resp.count("```") == 2          # LLMDFA asserts on failure here
        prog = ""
        if ok_fence:
            fenced += 1
            prog = resp[resp.find("```"): resp.rfind("```")].replace("`", "")
        verdict = "-"
        if prog:
            try:
                cp = subprocess.run(["python", "-c", prog], capture_output=True,
                                    text=True, timeout=60)
                if cp.stdout.strip():
                    verdict = cp.stdout.strip().splitlines()[-1]
                else:
                    err = (cp.stderr.strip().splitlines() or ["no stderr"])[-1]
                    verdict = "ERR"
                    errors.append(err)
            except Exception as e:
                verdict = "EXC:%s" % type(e).__name__
                errors.append(repr(e))
        if verdict in ("SAT", "UNSAT"):
            ran += 1
            right += (verdict == expected)
        print("  item %d: fenced=%s verdict=%s expected=%s (%.1fs)"
              % (i, ok_fence, verdict, expected, zlat[-1]), flush=True)

    report["phase4_z3"] = {
        "n": len(Z3_ITEMS),
        "well_fenced_rate": fenced / len(Z3_ITEMS),
        "first_try_run_rate": ran / len(Z3_ITEMS),
        "correct_verdict_rate": right / len(Z3_ITEMS),
        "avg_latency_s": sum(zlat) / len(zlat),
        "input_tokens": zin, "output_tokens": zout,
        "first_try_errors": errors,
    }

    report["llm_counters"] = {
        "calls": LLM.total_calls, "empty": LLM.empty_responses,
        "failed": LLM.failed_calls,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
