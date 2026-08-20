import csv
import os
import argparse
import subprocess
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import CWE_BENCH_JAVA_DIR, CODEQL_DB_PATH, PROJECT_SOURCE_CODE_DIR

def verify_java_installation(java_home):
    if not os.path.exists(java_home):
        raise Exception(f"JAVA_HOME directory does not exist: {java_home}")
    
    java_exe = os.path.join(java_home, 'bin', 'java')
    if not os.path.exists(java_exe):
        raise Exception(f"Java executable not found at: {java_exe}")
    
    javac_exe = os.path.join(java_home, 'bin', 'javac')
    if not os.path.exists(javac_exe):
        raise Exception(f"Javac executable not found at: {javac_exe}")

def verify_maven_installation(maven_path):
    if not os.path.exists(maven_path):
        raise Exception(f"Maven directory does not exist: {maven_path}")
    
    mvn_exe = os.path.join(maven_path, 'mvn')
    if not os.path.exists(mvn_exe):
        raise Exception(f"Maven executable not found at: {mvn_exe}")

def find_java_home(java_version, java_env_path):
    """
    Find the appropriate Java home directory based on version and available installations.
    
    Args:
        java_version (str): Version string from build_info.csv
        java_env_path (str): Base path for Java installations
    
    Returns:
        str: Path to the appropriate Java installation
    """
    if 'u' in java_version:
        # Handle Java 7 and 8 style versions (e.g., 8u202 -> jdk1.8.0_202)
        main_ver = java_version.split('u')[0]
        update_ver = java_version.split('u')[1]
        java_home = os.path.abspath(os.path.join(java_env_path, f"jdk1.{main_ver}.0_{update_ver}"))
    else:
        # Handle Java 9+ style versions
        # First try exact match (e.g., jdk-17)
        java_home = os.path.abspath(os.path.join(java_env_path, f"jdk-{java_version}"))
        
        if not os.path.exists(java_home):
            # Try finding a matching directory with a more specific version
            possible_dirs = [d for d in os.listdir(java_env_path) 
                           if d.startswith(f"jdk-{java_version}")]
            if possible_dirs:
                # Use the first matching directory
                java_home = os.path.abspath(os.path.join(java_env_path, possible_dirs[0]))
    
    return java_home

def setup_environment(row, java_env_path):
    env = os.environ.copy()
    
    # Set Maven path if available
    if row['mvn_version'] != 'n/a':
        maven_path = os.path.abspath(os.path.join(java_env_path, f"apache-maven-{row['mvn_version']}/bin"))
        verify_maven_installation(maven_path)
        env['PATH'] = f"{maven_path}:{env.get('PATH', '')}"
        print(f"Maven path set to: {maven_path}")
    
    # --- Reproduction fix (added) --------------------------------------------
    # build_info.csv records the toolchain that built the project, but CodeQL's
    # autobuild independently sniffs the source tree. Gradle-recorded projects
    # that still ship a pom.xml (e.g. zip4j) make autobuild invoke `mvn`, which
    # was never on PATH -> "Cannot run program mvn". Gradle was never added at
    # all. Append fallbacks AFTER any recorded version so the recorded one wins.
    fallback_bins = []
    for mv in ("3.9.8", "3.5.0", "3.2.1"):
        cand = os.path.abspath(os.path.join(java_env_path, f"apache-maven-{mv}/bin"))
        if os.path.exists(cand):
            fallback_bins.append(cand)
    gradle_recorded = row.get('gradle_version', 'n/a')
    gradle_order = [gradle_recorded] if gradle_recorded != 'n/a' else []
    gradle_order += ["8.9", "7.6.4", "6.8.2"]
    for gv in gradle_order:
        cand = os.path.abspath(os.path.join(java_env_path, f"gradle-{gv}/bin"))
        if os.path.exists(cand) and cand not in fallback_bins:
            fallback_bins.append(cand)
    if fallback_bins:
        env['PATH'] = f"{env.get('PATH', '')}:{':'.join(fallback_bins)}"
    # -------------------------------------------------------------------------

    # Find and set Java home
    java_version = row['jdk_version']
    java_home = find_java_home(java_version, java_env_path)
    
    verify_java_installation(java_home)
    env['JAVA_HOME'] = java_home
    print(f"JAVA_HOME set to: {java_home}")
    
    # Add Java binary to PATH
    env['PATH'] = f"{os.path.join(java_home, 'bin')}:{env.get('PATH', '')}"
    
    return env

def create_codeql_database(project_slug, env, db_base_path, sources_base_path):
    print("\nEnvironment variables for CodeQL database creation:")
    print(f"PATH: {env.get('PATH', 'Not set')}")
    print(f"JAVA_HOME: {env.get('JAVA_HOME', 'Not set')}")
    
    try:
        java_version = subprocess.check_output(['java', '-version'], 
                                            stderr=subprocess.STDOUT, 
                                            env=env).decode()
        print(f"\nJava version check:\n{java_version}")
    except subprocess.CalledProcessError as e:
        print(f"Error checking Java version: {e}")
        raise
    
    database_path = os.path.abspath(os.path.join(db_base_path, project_slug))
    source_path = os.path.abspath(os.path.join(sources_base_path, project_slug))
    
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    
    command = [
        "codeql", "database", "create",
        database_path,
        "--source-root", source_path,
        "--language", "java",
        "--overwrite"
    ]

    # --- Reproduction fix (added) --------------------------------------------
    # data/build_info.csv is the upstream record; build-info/<slug>.json is what
    # OUR build actually succeeded with (build_one.py walks a 10-attempt ladder
    # and dumps the winner). Prefer the latter. For gradle-built projects give
    # CodeQL the explicit build command, mirroring build_one.py, because
    # autobuild picks maven whenever a pom.xml exists and then fails.
    bi = os.path.join(os.path.dirname(os.path.dirname(sources_base_path)),
                      "build-info", f"{project_slug}.json")
    if os.path.exists(bi):
        import json as _json
        actual = _json.load(open(bi))
        if "gradle" in actual:
            command += ["--command", "gradle build --parallel"]
            print(f"Using recorded gradle build command for {project_slug}")
        elif "gradlew" in actual:
            command += ["--command", "./gradlew --no-daemon -S -Dorg.gradle.dependency.verification=off build"]
            print(f"Using recorded gradlew build command for {project_slug}")
    # -------------------------------------------------------------------------
    
    try:
        print(f"Creating database at: {database_path}")
        print(f"Using source path: {source_path}")
        print(f"Using JAVA_HOME: {env.get('JAVA_HOME', 'Not set')}")
        subprocess.run(command, env=env, check=True)
        print(f"Successfully created CodeQL database for {project_slug}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating CodeQL database for {project_slug}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Create CodeQL databases for cwe-bench-java projects')
    parser.add_argument('--project', help='Specific project slug', default=None)
    parser.add_argument('--db-path', help='Base path for storing CodeQL databases', default=CODEQL_DB_PATH)
    parser.add_argument('--sources-path', help='Base path for project sources', default=PROJECT_SOURCE_CODE_DIR)
    parser.add_argument('--cwe-bench-java-path', help='Base path to cwe-bench-java', default=CWE_BENCH_JAVA_DIR)
    args = parser.parse_args()
    
    cwe_bench_java_path = os.path.abspath(args.cwe_bench_java_path)
    csv_path = os.path.join(cwe_bench_java_path, "data", "build_info.csv")
    java_env_path = os.path.join(cwe_bench_java_path, "java-env")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        projects = list(reader)
    
    if args.project:
        project = next((p for p in projects if p['project_slug'] == args.project), None)
        if project:
            env = setup_environment(project, java_env_path)
            create_codeql_database(project['project_slug'], env, args.db_path, args.sources_path)
        else:
            print(f"Project {args.project} not found in CSV file")
    else:
        for project in projects:
            env = setup_environment(project, java_env_path)
            create_codeql_database(project['project_slug'], env, args.db_path, args.sources_path)

if __name__ == "__main__":
    main()