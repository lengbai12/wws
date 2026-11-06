#!/usr/bin/env python3
"""Fetch the replay schema this tool needs, pinned to one upstream commit.

Downloads, into a cache directory outside version control:

* the upstream parser source (``replay_unpack``) at a pinned commit
* the game data for one client build, placed where the parser expects it
* curated enrichment tables (consumable type ids, projectiles)

Game data is Wargaming property and changes every patch, so it is fetched on
demand rather than vendored. Re-run with ``--build`` for a new client build.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UPSTREAM_REPO = "Monstrofil/replays_unpack"
ENRICH_REPO = "toalba/wows-render-gamedata"
# Pinned so the schema layer cannot change under us between runs.
DEFAULT_COMMIT = "fd86f972b0ac9833616d5104e86229da6d2f5088"
DEFAULT_BUILD = "15.8.0"
ENRICH_FILES = ("consumable_type_ids.json", "projectiles.json")
DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache"


def _get_once(url: str, timeout: int) -> bytes:
    # curl first. On this host urllib's transport can hang far past its own
    # timeout (measured: the same URL that curl fetches in about a second, and
    # that the Python API serves instantly, never returns through urllib), so
    # trying urllib first made a 185-file download take hours with the fallback
    # unreachable until each attempt timed out.
    try:
        result = subprocess.run(
            ["curl", "-sS", "--fail", "-m", str(timeout), url],
            capture_output=True, check=True,
        )
        return result.stdout
    except (OSError, subprocess.CalledProcessError):
        request = urllib.request.Request(
            url, headers={"User-Agent": "wws-replay-tools"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()


def _get(url: str, timeout: int = 60, attempts: int = 5) -> bytes:
    """Fetch with backoff; the API and CDN both fail intermittently."""
    last = None
    for attempt in range(attempts):
        try:
            return _get_once(url, timeout)
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last = exc
            time.sleep(min(2 ** attempt, 15))
    raise SystemExit(f"giving up on {url} after {attempts} attempts: {last}")


def _tree(repo: str, commit: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"
    return [entry for entry in json.loads(_get(url))["tree"] if entry["type"] == "blob"]


def _download_all(jobs: list[tuple[str, Path]], workers: int = 10) -> int:
    def one(job: tuple[str, Path]) -> bool:
        url, dest = job
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(_get(url))
            return True
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  failed {url}: {exc}", file=sys.stderr)
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(one, jobs))


def fetch(build: str, commit: str, cache: Path, refresh: bool) -> Path:
    version_dir = build.replace(".", "_")
    root = cache / f"{commit[:8]}_{version_dir}"
    parser_root = root / "replay_unpack"
    if root.exists() and not refresh:
        print(f"cache hit: {root}")
        return root

    if root.exists():
        shutil.rmtree(root)
    raw = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{commit}"

    # One tree call: the unauthenticated API is rate limited.
    tree = _tree(UPSTREAM_REPO, commit)
    sources = [e["path"] for e in tree
               if e["path"].startswith("replay_unpack/") and "/versions/" not in e["path"]]
    print(f"fetching parser source at {commit[:8]}: {len(sources)} files ...")
    jobs = [(f"{raw}/{p}", root / p) for p in sources]

    prefix = f"replay_unpack/clients/wows/versions/{version_dir}/"
    game_data = [e["path"] for e in tree if e["path"].startswith(prefix)]
    if not game_data:
        raise SystemExit(f"upstream has no game data for build {build} (expected {prefix})")
    print(f"fetching game data for {build}: {len(game_data)} files ...")
    jobs += [(f"{raw}/{p}", parser_root.parent / p) for p in game_data]

    done = _download_all(jobs)
    print(f"  {done}/{len(jobs)} files")

    print("fetching enrichment tables ...")
    enrich = root / "enrich"
    enrich.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{ENRICH_REPO}/main/data"
    for name in ENRICH_FILES:
        try:
            (enrich / name).write_bytes(_get(f"{base}/{name}"))
        except Exception as exc:  # noqa: BLE001 - enrichment is optional
            print(f"  optional enrichment {name} unavailable: {exc}", file=sys.stderr)

    marker = {"commit": commit, "build": build, "upstream": UPSTREAM_REPO,
              "enrichment": ENRICH_REPO}
    (root / "schema.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(f"ready: {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default=DEFAULT_BUILD,
                        help="client build, e.g. 15.8.0 (default: %(default)s)")
    parser.add_argument("--commit", default=DEFAULT_COMMIT,
                        help="upstream commit to pin (default: %(default)s)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    fetch(args.build, args.commit, args.cache, args.refresh)


if __name__ == "__main__":
    main()
