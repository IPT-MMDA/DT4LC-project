#!/usr/bin/env python3
"""Delete stale versions of the DT4LC container packages on GHCR.

A package version is one image manifest, and every tag that points at it belongs to that
version, so deleting a version removes all of its tags. This script lists the versions of
``ghcr.io/ipt-mmda/dt4lc/backend`` and ``.../frontend`` and deletes only those whose
every tag matches one of the patterns given; untagged versions and versions carrying any other
tag are kept. Without ``--delete`` it only prints what it would do.

    GHCR_TOKEN=<classic token with read:packages and delete:packages> \\
        python3 scripts/ghcr_prune.py '^1\\.0\\.0-' '^0\\.2\\.0'            # dry run
    GHCR_TOKEN=... python3 scripts/ghcr_prune.py --delete '^1\\.0\\.0-' '^0\\.2\\.0'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ORG = "IPT-MMDA"
PACKAGES = ["dt4lc/backend", "dt4lc/frontend"]
API = "https://api.github.com"


def request(method: str, url: str, token: str) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def versions(package: str, token: str) -> list[dict]:
    name = urllib.parse.quote(package, safe="")
    out: list[dict] = []
    page = 1
    while True:
        url = f"{API}/orgs/{ORG}/packages/container/{name}/versions?per_page=100&page={page}"
        status, body = request("GET", url, token)
        if status != 200:
            sys.exit(f"{package}: HTTP {status} listing versions: {body[:200]!r}")
        batch = json.loads(body)
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "patterns", nargs="+", help="regular expressions; a version is stale when every tag matches one"
    )
    parser.add_argument("--delete", action="store_true", help="delete the stale versions (default: list them)")
    args = parser.parse_args()
    token = os.environ.get("GHCR_TOKEN")
    if not token:
        sys.exit("set GHCR_TOKEN to a token with read:packages and delete:packages")
    patterns = [re.compile(p) for p in args.patterns]

    for package in PACKAGES:
        stale = []
        for v in versions(package, token):
            tags = v.get("metadata", {}).get("container", {}).get("tags", [])
            if tags and all(any(p.search(t) for p in patterns) for t in tags):
                stale.append((v["id"], tags, v.get("created_at", "")[:10]))
        print(f"{package}: {len(stale)} stale version(s)")
        for vid, tags, created in stale:
            print(f"  {created}  {' '.join(tags)}")
            if args.delete:
                name = urllib.parse.quote(package, safe="")
                status, body = request("DELETE", f"{API}/orgs/{ORG}/packages/container/{name}/versions/{vid}", token)
                print("    deleted" if status == 204 else f"    HTTP {status}: {body[:120]!r}")
    if not args.delete:
        print("dry run; add --delete to remove them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
