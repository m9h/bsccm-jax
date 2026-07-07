#!/usr/bin/env python
"""Resumable BSCCM download from Dryad via OAuth2 client-credentials.

Credentials are NEVER passed on the command line or hard-coded. Provide them as
either environment variables:

    DRYAD_CLIENT_ID=...  DRYAD_CLIENT_SECRET=...

or an env-style file at ~/.config/dryad/credentials (chmod 600):

    DRYAD_CLIENT_ID=xxxxxxxx
    DRYAD_CLIENT_SECRET=yyyyyyyy

The Dryad OAuth token is short-lived, so a 228 GB pull can outlive it. This
wrapper re-mints a fresh token on every attempt and relies on download_dataset()
skipping any file already present at full size — so it resumes at file
granularity across token expiry, network drops, or a killed process.

Usage:
    uv run python scripts/dryad_download.py --location /mnt/truenas/bsccm/     # tiny (~490 MB)
    uv run python scripts/dryad_download.py --location /mnt/truenas/bsccm/ --full            # ~196 GB
    uv run python scripts/dryad_download.py --location /mnt/truenas/bsccm/ --full --coherent # +24 GB
"""

import argparse
import os
import pathlib
import sys
import time

import requests
from bsccm import download_dataset

TOKEN_URL = "https://datadryad.org/oauth/token"
CREDS_FILE = pathlib.Path.home() / ".config" / "dryad" / "credentials"


def load_creds():
    cid = os.environ.get("DRYAD_CLIENT_ID")
    sec = os.environ.get("DRYAD_CLIENT_SECRET")
    if not (cid and sec) and CREDS_FILE.exists():
        for line in CREDS_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("DRYAD_CLIENT_ID="):
                cid = line.split("=", 1)[1].strip()
            elif line.startswith("DRYAD_CLIENT_SECRET="):
                sec = line.split("=", 1)[1].strip()
    if not (cid and sec):
        sys.exit(f"No Dryad creds. Set DRYAD_CLIENT_ID/DRYAD_CLIENT_SECRET or write {CREDS_FILE}")
    return cid, sec


def get_token(cid, sec):
    r = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": cid, "client_secret": sec,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="data/", help="download target dir")
    ap.add_argument("--full", action="store_true", help="full dataset (default: tiny subset)")
    ap.add_argument("--coherent", action="store_true", help="single-LED coherent variant")
    ap.add_argument("--mnist", action="store_true", help="BSCCMNIST variant")
    ap.add_argument("--max-retries", type=int, default=100)
    args = ap.parse_args()

    cid, sec = load_creds()
    pathlib.Path(args.location).mkdir(parents=True, exist_ok=True)
    tiny = not args.full

    for attempt in range(1, args.max_retries + 1):
        try:
            token = get_token(cid, sec)
            print(f"[attempt {attempt}] token acquired (…{token[-6:]}); starting/resuming download")
            path = download_dataset(location=args.location, coherent=args.coherent,
                                    tiny=tiny, mnist=args.mnist, token=token)
            print("COMPLETE:", path)
            return
        except KeyboardInterrupt:
            sys.exit("interrupted by user")
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"[attempt {attempt}] {type(e).__name__}: {e} — retrying in {wait}s")
            time.sleep(wait)
    sys.exit("exhausted retries")


if __name__ == "__main__":
    main()
