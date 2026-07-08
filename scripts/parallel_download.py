"""Parallel BSCCM download from Dryad — ~24x faster than the sequential downloader.

Dryad throttles per *connection* (~30 MB/s single-stream), not per IP, so fetching
the ~50 tarball chunks concurrently hits ~90 MB/s aggregate — the full 197 GB main
variant in ~35 min instead of hours. Chunks resume (skip if already complete), the
OAuth token is refreshed on expiry, then chunks are concatenated and extracted.

    DRYAD_CLIENT_ID=... DRYAD_CLIENT_SECRET=... \
      python scripts/parallel_download.py --dest /mnt/t9/bsccm --workers 8 --extract

Variants: --variant {main,coherent,mnist}.
"""

import argparse
import concurrent.futures as cf
import shutil
import tarfile
import threading
import time
from pathlib import Path

import requests

from dryad_download import get_token, load_creds

BASE = "https://datadryad.org"
DOI = "doi%3A10.5061%2Fdryad.sxksn038s"
PREFIX = {"main": "BSCCM.tar.gz_chunk",
          "coherent": "BSCCM-coherent.tar.gz_chunk",
          "mnist": "BSCCMNIST.tar.gz_chunk"}

_tok = {"v": None}
_lock = threading.Lock()


def _token():
    with _lock:
        if _tok["v"] is None:
            _tok["v"] = get_token(*load_creds())
        return _tok["v"]


def _refresh():
    with _lock:
        _tok["v"] = get_token(*load_creds())
        return _tok["v"]


def list_files():
    h = {"Authorization": f"Bearer {_token()}", "accept": "application/json"}
    v = requests.get(f"{BASE}/api/v2/datasets/{DOI}/versions", headers=h, timeout=30).json()
    vid = v["_embedded"]["stash:versions"][-1]["_links"]["self"]["href"].split("/")[-1]
    files, url = [], f"{BASE}/api/v2/versions/{vid}/files"
    while url:
        r = requests.get(url, headers=h, timeout=30).json()
        files += r["_embedded"]["stash:files"]
        nxt = r["_links"].get("next")
        url = (BASE + nxt["href"]) if nxt else None
    return files


def fetch(f, dest, retries=6):
    path = dest / f["path"]
    if path.exists() and path.stat().st_size == f["size"]:
        return f["size"]                                      # resume: already done
    url = BASE + f["_links"]["stash:download"]["href"]
    for attempt in range(retries):
        try:
            tok = _token()
            with requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                              stream=True, timeout=120) as r:
                if r.status_code == 401:
                    _refresh(); continue
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as out:
                    for c in r.iter_content(1 << 22):
                        out.write(c)
                tmp.rename(path)
            if path.stat().st_size == f["size"]:
                return f["size"]
        except Exception:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"failed: {f['path']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/mnt/t9/bsccm")
    ap.add_argument("--variant", default="main", choices=list(PREFIX))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest); dest.mkdir(parents=True, exist_ok=True)
    chunks = sorted((f for f in list_files() if f["path"].startswith(PREFIX[args.variant])),
                    key=lambda x: x["path"])
    total = sum(f["size"] for f in chunks)
    print(f"{args.variant}: {len(chunks)} chunks, {total/1e9:.1f} GB -> {dest} "
          f"({args.workers} parallel)")

    t0 = time.time(); done = [0]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, f, dest): f for f in chunks}
        for fut in cf.as_completed(futs):
            done[0] += fut.result()
            el = time.time() - t0
            print(f"  {done[0]/1e9:6.1f}/{total/1e9:.1f} GB  {done[0]/el/1e6:5.1f} MB/s  "
                  f"({100*done[0]/total:.0f}%)", flush=True)

    if args.extract:
        # space-efficient: stream chunks into `tar -xz`, deleting each chunk once
        # it's been fed in, so peak disk stays ~= one copy of the data (not two).
        import subprocess
        out_dir = dest / "extracted"; out_dir.mkdir(exist_ok=True)
        names = " ".join(f'"{dest / f["path"]}"' for f in chunks)
        print(f"stream-extracting {len(chunks)} chunks -> {out_dir} (deleting as consumed)")
        pipe = f'for c in {names}; do cat "$c" && rm -f "$c"; done | tar -xzf - -C "{out_dir}"'
        subprocess.run(pipe, shell=True, check=True, executable="/bin/bash")
        print(f"DONE in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    else:
        combined = dest / PREFIX[args.variant].split("_chunk")[0]
        print(f"combining -> {combined}")
        with open(combined, "wb") as out:
            for f in chunks:
                with open(dest / f["path"], "rb") as p:
                    shutil.copyfileobj(p, out, 1 << 24)
        print(f"DONE in {(time.time()-t0)/60:.1f} min -> {combined}")


if __name__ == "__main__":
    main()
