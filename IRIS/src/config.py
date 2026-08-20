import os

IRIS_ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")

# CODEQL_DIR should be the path of the patched version of CodeQL provided as a download in the releases section for Iris.
CODEQL_DIR = f"{IRIS_ROOT_DIR}/codeql"

# CODEQL_DB_PATH is the path to the directory that contains CodeQL databases.
CODEQL_DB_PATH = f"{IRIS_ROOT_DIR}/data/codeql-dbs"

# PROJECT_SOURCE_CODE_DIR contains the Java projects. 
PROJECT_SOURCE_CODE_DIR = f"{IRIS_ROOT_DIR}/data/cwe-bench-java/project-sources"

# PACKAGE_MODULES_PATH contains each project's internal modules. 
PACKAGE_MODULES_PATH = f"{IRIS_ROOT_DIR}/data/cwe-bench-java/package-names"

# OUTPUT_DIR is where the results from running Iris are stored.
OUTPUT_DIR = f"{IRIS_ROOT_DIR}/output"

# ALL_METHOD_INFO_DIR  
ALL_METHOD_INFO_DIR = f"{IRIS_ROOT_DIR}/data/cwe-bench-java/data/fix_info.csv"

# CVES_MAPPED_W_COMMITS_DIR is the path to project_info.csv, which contains the mapping of vulnerabilities to projects in cwe-bench-java. 
CVES_MAPPED_W_COMMITS_DIR = f"{IRIS_ROOT_DIR}/data/cwe-bench-java/data/project_info.csv"

# Path to cwe-bench-java directory submodule.
CWE_BENCH_JAVA_DIR = f"{IRIS_ROOT_DIR}/data/cwe-bench-java"