import argparse
import os
import sys
import subprocess as sp
import pandas as pd

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
NEUROSYMSA_ROOT_DIR = os.path.abspath(f"{THIS_SCRIPT_DIR}/../")
sys.path.append(NEUROSYMSA_ROOT_DIR)

from src.config import CVES_MAPPED_W_COMMITS_DIR
from src.queries import QUERIES

def collect_projects_for_query(query, cwe_id, all_cves_with_commit, all_project_tags):
    for (_, proj_row) in all_cves_with_commit.iterrows():
        # Check relevance
        if f"CWE-{cwe_id}" not in proj_row["cwe_id"].split(";"):
            continue
        cve_id = proj_row["cve_id"]
        relevant_project_tag = all_project_tags[all_project_tags["cve_id"] == cve_id]
        if len(relevant_project_tag) == 0:
            continue
        project_name = relevant_project_tag.iloc[0]["project_slug"]
        yield project_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="cwe-022wLLM")
    parser.add_argument("--llm", type=str, default="gpt-4")
    parser.add_argument("--run-id", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--label-api-batch-size", type=int, default=30)
    parser.add_argument("--label-func-param-batch-size", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=3)
    parser.add_argument("--no-summary-model", action="store_true")
    parser.add_argument("--use-exhaustive-qll", action="store_true")
    parser.add_argument("--skip-huge-project", action="store_true")
    parser.add_argument("--skip-huge-project-num-apis-threshold", type=int, default=3000)
    parser.add_argument("--skip-project", type=str, nargs="+")
    parser.add_argument("--filter-project", type=str, nargs="+")
    parser.add_argument("--skip-posthoc-filter", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--filter-by-module", action="store_true")
    parser.add_argument("--posthoc-filtering-skip-fp", action="store_true")
    parser.add_argument("--posthoc-filtering-rerun-skipped-fp", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-api-candidates", action="store_true")
    parser.add_argument("--overwrite-func-param-candidates", action="store_true")
    parser.add_argument("--overwrite-labelled-apis", action="store_true")
    parser.add_argument("--overwrite-llm-cache", action="store_true")
    parser.add_argument("--overwrite-labelled-func-param", action="store_true")
    parser.add_argument("--overwrite-cwe-query-result", action="store_true")
    parser.add_argument("--overwrite-posthoc-filter", action="store_true")
    parser.add_argument("--overwrite-debug-info", action="store_true")
    parser.add_argument("--debug-source", action="store_true")
    parser.add_argument("--debug-sink", action="store_true")
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    query = args.query
    if query not in QUERIES:
        print(f"Unknown query {query}")
    if "cwe_id_tag" not in QUERIES[query]:
        print(f"Not a CWE related query: {query}")
    cwe_id = QUERIES[query]["cwe_id"]

    all_cves_with_commit = pd.read_csv(CVES_MAPPED_W_COMMITS_DIR)
    all_project_tags = all_cves_with_commit.dropna(subset=["project_slug", "cve_id", "github_tag"])

    relevant_projects = list(collect_projects_for_query(query, cwe_id, all_cves_with_commit, all_project_tags))

    for (i, project) in enumerate(relevant_projects):
        print("===========================================")
        print(f"[{i + 1}/{len(relevant_projects)}] STARTING RUNNING ON PROJECT: {project}")

        # Skip if not desired
        if args.skip_project is not None:
            need_skip = False
            for skip_project_filter in args.skip_project:
                if skip_project_filter in project:
                    need_skip = True; break
            if need_skip:
                continue

        if args.filter_project is not None:
            need_skip = True
            for filter_project_filter in args.filter_project:
                if filter_project_filter in project:
                    need_skip = False; break
            if need_skip:
                continue

        # Generate the command
        command = [
            "python", f"{THIS_SCRIPT_DIR}/neusym_vul.py",
            project,
            "--query", query,
            "--llm", args.llm,
            "--run-id", args.run_id,
            "--seed", str(args.seed),
            "--label-api-batch-size", str(args.label_api_batch_size),
            "--num-threads", str(args.num_threads),
        ]

        # Adding store_true arguments
        if args.no_summary_model: command += ["--no-summary-model"]
        if args.use_exhaustive_qll: command += ["--use-exhaustive-qll"]
        if args.skip_huge_project: command += ["--skip-huge-project", "--skip-huge-project-num-apis-threshold", str(args.skip_huge_project_num_apis_threshold)]
        if args.skip_posthoc_filter: command += ["--skip-posthoc-filter"]
        if args.skip_evaluation: command += ["--skip-evaluation"]
        if args.posthoc_filtering_skip_fp: command += ["--posthoc-filtering-skip-fp"]
        if args.posthoc_filtering_rerun_skipped_fp: command += ["--posthoc-filtering-rerun-skipped-fp"]
        if args.evaluation_only: command += ["--evaluation-only"]
        if args.filter_by_module: command += ["--filter-by-module"]

        # Overwrites
        if args.overwrite: command += ["--overwrite"]
        if args.overwrite_api_candidates: command += ["--overwrite-api-candidates"]
        if args.overwrite_func_param_candidates: command += ["--overwrite-func-param-candidates"]
        if args.overwrite_labelled_apis: command += ["--overwrite-labelled-apis"]
        if args.overwrite_llm_cache: command += ["--overwrite-llm-cache"]
        if args.overwrite_labelled_func_param: command += ["--overwrite-labelled-func-param"]
        if args.overwrite_cwe_query_result: command += ["--overwrite-cwe-query-result"]
        if args.overwrite_posthoc_filter: command += ["--overwrite-posthoc-filter"]
        if args.debug_source: command += ["--debug-source"]
        if args.debug_sink: command += ["--debug-sink"]
        if args.test_run: command += ["--test-run"]

        # Run the command
        sp.run(command)
