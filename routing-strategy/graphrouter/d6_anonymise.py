#!/usr/bin/env python3
"""D6 -- hide project identity from the code paths, keep the logic.

The memorisation control for D1. Every CVE in CWE-Bench-Java predates Qwen3's
training data, so Qwen3-32B's within-project AUC of 0.868 could come from
RECOGNISING known-vulnerable code rather than reading it. This rewrites every
path so the project cannot be recognised, then D6's probe asks the same question.

Which names to hide is decided from data, not by hand. All 120 CWE-Bench-Java
source trees are tokenised; a name that appears in >= MIN_REPOS different
repositories is ordinary Java vocabulary (String, File, exec, getParameter) and
is KEPT, because it carries meaning a real reviewer needs. A name confined to
fewer repositories is project vocabulary (HashTrie, StringSubstitutor,
FeatureStore) and is RENAMED. Versions of one repository (antisamy 2016/2017,
the four DSpace CVEs) count as one repository.

Two levels:
  a  "renamed"  project names -> Cls1 / name1 / CONST1, consistently within a
                path so the dataflow still reads; file paths and line numbers
                -> F1.java; comments and string literals otherwise untouched
  b  "strict"   keeps only names used in >= STRICT_REPOS repositories (the
                everyday Java vocabulary), so even widely used library classes
                that identify a CVE (StringSubstitutor for Text4Shell) are
                renamed; + every comment removed (except the CodeQL alert header,
                which states the task and is identical across a project's
                paths) + string literal contents -> "s1", "s2"

Output: one units file per level, same unit_id / gold_label / project as the
original, so the probe and the analysis join on unit_id.
"""
import argparse, collections, json, os, re
from concurrent.futures import ProcessPoolExecutor

JAVA_KEYWORDS = set("""
abstract assert boolean break byte case catch char class const continue default
do double else enum extends final finally float for goto if implements import
instanceof int interface long native new package private protected public return
short static strictfp super switch synchronized this throw throws transient try
void volatile while var record yield sealed permits non true false null
""".split())

TOKEN = re.compile(r'''
    (?P<lcomment>//[^\n]*)
  | (?P<bcomment>/\*.*?\*/)
  | (?P<string>"(?:\\.|[^"\\\n])*")
  | (?P<char>'(?:\\.|[^'\\\n])*')
  | (?P<ident>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<other>\s+|.)
''', re.S | re.X)
IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
HEADER = re.compile(r"^// (\S+\.java):\d+\s+\S+\s*$")      # "// path/File.java:184  put"


def repo_of(project_slug: str) -> str:
    """ESAPI__esapi-java-legacy_CVE-2022-24891_2.2.3.1 -> ESAPI__esapi-java-legacy"""
    return re.split(r"_CVE-", project_slug)[0]


def names_in_tree(root: str) -> set:
    out = set()
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".java"):
                try:
                    out.update(IDENT.findall(open(os.path.join(dp, f), errors="ignore").read()))
                except OSError:
                    pass
    return out


def build_vocab(sources: str, min_repos: int, cache: str) -> dict:
    """-> {name: number of repositories using it}, for names used in >= min_repos."""
    if os.path.exists(cache):
        return json.load(open(cache))
    projects = sorted(os.listdir(sources))
    with ProcessPoolExecutor(16) as ex:
        per_project = dict(zip(projects, ex.map(names_in_tree, [os.path.join(sources, p) for p in projects])))
    per_repo = collections.defaultdict(set)
    for p, names in per_project.items():
        per_repo[repo_of(p)] |= names
    count = collections.Counter(n for names in per_repo.values() for n in names)
    keep = {n: c for n, c in count.items() if c >= min_repos}
    print(f"{len(projects)} projects, {len(per_repo)} repositories, {len(count)} distinct names; "
          f"{len(keep)} appear in >= {min_repos} repositories")
    json.dump(keep, open(cache, "w"))
    return keep


def anonymise(code: str, keep: set, strict: bool) -> tuple[str, int, int]:
    lines = code.split("\n")
    # the CodeQL alert header (everything before the first method) states the
    # task and is identical across a project's paths; keep it verbatim
    first = next((i for i, l in enumerate(lines) if HEADER.match(l)), 0)
    head, body = lines[:first], lines[first:]
    files, names, strings = {}, {}, {}
    counters = collections.Counter()
    renamed = kept = 0

    def placeholder(tok):
        if tok not in names:
            kind = ("CONST" if tok.isupper() and len(tok) > 1 else
                    "Cls" if tok[0].isupper() else "name")
            counters[kind] += 1
            names[tok] = f"{kind}{counters[kind]}"
        return names[tok]

    def rename_ident(tok):
        nonlocal renamed, kept
        if tok in JAVA_KEYWORDS or tok in keep:
            kept += 1
            return tok
        renamed += 1
        return placeholder(tok)

    out = []
    for line in body:
        m = HEADER.match(line)
        if m:                                   # file path + line + method -> F<n>.java
            files.setdefault(m.group(1), f"F{len(files) + 1}.java")
            out.append(f"// {files[m.group(1)]}")
            continue
        out.append(line)
    text = "\n".join(out)

    pieces = []
    for m in TOKEN.finditer(text):
        kind, tok = m.lastgroup, m.group()
        if kind in ("lcomment", "bcomment"):
            if strict and not re.fullmatch(r"// F\d+\.java", tok):
                continue
            pieces.append(IDENT.sub(lambda x: rename_ident(x.group()) if x.group() in names or
                                    (x.group() not in keep and x.group() not in JAVA_KEYWORDS and
                                     not x.group().islower()) else x.group(), tok)
                          if not re.fullmatch(r"// F\d+\.java", tok) else tok)
        elif kind == "string":
            if strict:
                strings.setdefault(tok, f'"s{len(strings) + 1}"')
                pieces.append(strings[tok])
            else:
                pieces.append(tok)
        elif kind == "ident":
            pieces.append(rename_ident(tok))
        else:
            pieces.append(tok)
    text = "".join(pieces)
    if strict:                                  # drop lines emptied by comment removal
        text = "\n".join(l for l in text.split("\n") if l.strip())
    return "\n".join(head + [text]), renamed, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=os.path.expanduser("~/nimit/graphrouter/data/units_paths.jsonl"))
    ap.add_argument("--sources", default=os.path.expanduser(
        "~/nimit/vuln-pred-results/IRIS/data/cwe-bench-java/project-sources"))
    ap.add_argument("--min-repos", type=int, default=5, help="level a: keep names in >= this many repos")
    ap.add_argument("--strict-repos", type=int, default=40, help="level b: keep names in >= this many repos")
    ap.add_argument("--vocab-cache", default=os.path.expanduser("~/nimit/graphrouter/data/d6_vocab_counts.json"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/nimit/graphrouter/data"))
    a = ap.parse_args()

    counts = build_vocab(a.sources, a.min_repos, a.vocab_cache)
    units = [json.loads(l) for l in open(a.units) if l.strip()]
    for level, strict, thr in (("anonA", False, a.min_repos), ("anonB", True, a.strict_repos)):
        keep = {n for n, c in counts.items() if c >= thr}
        tot_r = tot_k = 0
        path = os.path.join(a.out_dir, f"units_paths_{level}.jsonl")
        with open(path, "w") as fh:
            for u in units:
                code, r, k = anonymise(u["code"], keep, strict)
                tot_r += r; tot_k += k
                fh.write(json.dumps({**u, "code": code}) + "\n")
        print(f"{level}: keep names in >= {thr} repos ({len(keep)} names); {len(units)} paths -> {path}; "
              f"identifiers renamed {tot_r}, kept {tot_k} "
              f"({100 * tot_r / max(tot_r + tot_k, 1):.0f}% renamed)")


if __name__ == "__main__":
    main()
