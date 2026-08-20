import argparse
import os
import sys
import subprocess as sp
import pandas as pd

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
NEUROSYMSA_ROOT_DIR = os.path.abspath(f"{THIS_SCRIPT_DIR}/../../")
sys.path.append(NEUROSYMSA_ROOT_DIR)

from src.config import CVES_MAPPED_W_COMMITS_DIR, CVE_REPO_TAGS_DIR


from src.queries import QUERIES

def collect_projects_for_query(query, cwe_id, all_cves_with_commit, all_project_tags):
    for (_, proj_row) in all_cves_with_commit.iterrows():
        # Check relevance
        if cwe_id not in proj_row["cwe"].split(";"):
            continue
        cve_id = proj_row["cve"]
        relevant_project_tag = all_project_tags[all_project_tags["cve"] == cve_id]
        if len(relevant_project_tag) == 0:
            continue
        project_name = relevant_project_tag.iloc[0]["project"]
        yield project_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, default="cwe-022wCodeQL")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    query = args.query
    if query not in QUERIES:
        print(f"Unknown query {query}")
    if "cwe_id_tag" not in QUERIES[query]:
        print(f"Not a CWE related query: {query}")
    cwe_id = QUERIES[query]["cwe_id_tag"]

    all_cves_with_commit = pd.read_csv(CVES_MAPPED_W_COMMITS_DIR).dropna(subset=["cwe", "cve", "commits"])
    all_project_tags = pd.read_csv(CVE_REPO_TAGS_DIR).dropna(subset=["project", "cve", "tag"])

    relevant_projects = list(collect_projects_for_query(query, cwe_id, all_cves_with_commit, all_project_tags))

    for (i, project) in enumerate(relevant_projects):
        print("===========================================")
        print(f"[{i + 1}/{len(relevant_projects)}] STARTING RUNNING ON PROJECT: {project}")

        # Generate the command
        command = ["python", f"{THIS_SCRIPT_DIR}/codeql_vul.py", project, "--query", query]
        if args.evaluation_only: command += ["--evaluation-only"]
        if args.overwrite: command += ["--overwrite"]

        # Run the command
        sp.run(command)
