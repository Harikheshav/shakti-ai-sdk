#!/usr/bin/env python3
"""Downloads with checksums: models, datasets and the sources a framework fetches at build time.

Every entry of third_party.lock (sources) and models/manifest.yaml (models) is URL + sha256; files go to a cache
(SHAKTI_AI_CACHE, default ~/.cache/shakti-ai-sdk), never into the repository.  A wrong checksum is an error.

usage: tools/fetch.py NAME [NAME ...]     fetch these entries (prints the cached path of each)
       tools/fetch.py --list              the entries and whether they are cached
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("SHAKTI_AI_CACHE", os.path.expanduser("~/.cache/shakti-ai-sdk"))


def entries():
    """{name: {url, sha256, file}} from third_party.lock and models/manifest.yaml (simple 'key: value' blocks)."""
    out = {}
    for f in (os.path.join(ROOT, "third_party.lock"), os.path.join(ROOT, "models", "manifest.yaml")):
        if not os.path.isfile(f):
            continue
        cur = None
        for line in open(f):
            line = line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith(" ") and line.endswith(":"):
                cur = out.setdefault(line[:-1].strip(), {})
            elif cur is not None and ":" in line:
                k, v = line.strip().split(":", 1)
                cur[k.strip()] = v.strip().strip('"')
    for n, e in out.items():
        if not e.get("url") or not e.get("sha256"):
            sys.exit(f"{n}: needs url and sha256")
        e.setdefault("file", os.path.basename(e["url"].split("?")[0]))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, e):
    d = os.path.join(CACHE, e["sha256"][:16])
    p = os.path.join(d, e["file"])
    if os.path.isfile(p) and sha256(p) == e["sha256"]:
        return p
    os.makedirs(d, exist_ok=True)
    tmp = p + ".part"
    urllib.request.urlretrieve(e["url"], tmp)
    got = sha256(tmp)
    if got != e["sha256"]:
        os.remove(tmp)
        sys.exit(f"{name}: sha256 {got} does not match the lock ({e['sha256']})")
    os.replace(tmp, p)
    return p


def main(argv):
    es = entries()
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        for n, e in sorted(es.items()):
            p = os.path.join(CACHE, e["sha256"][:16], e["file"])
            print(f"{n:30s} {'cached' if os.path.isfile(p) else '-':7s} {e['url']}")
        return 0
    for n in argv:
        if n not in es:
            sys.exit(f"unknown entry '{n}'")
        print(fetch(n, es[n]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
