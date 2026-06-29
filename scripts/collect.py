#!/usr/bin/env python3
"""collect.py — Data acquisition.

Single responsibility: get the raw Rakuten data from a configured source into
data/raw/ in the canonical layout, idempotently, with verification and an audit
manifest. It is the ONLY part of the pipeline that knows where data comes from;
nothing downstream cares.

Source may be:
  * --source <local .zip>    -> extracted + normalized
  * --source <local dir>     -> contents normalized into data/raw/
  * --source <http(s) URL>   -> downloaded then extracted
  * --from-ens               -> authenticated download from challengedata.ens.fr
                                using ENS_USERNAME / ENS_PASSWORD from the env

Canonical result:
  data/raw/X_train_update.csv
  data/raw/Y_train_CVw08PX.csv
  data/raw/images/image_train/<image files>
  data/raw/manifest.json

Examples:
  python scripts/collect.py --source /home/me/rakuten/data      # local folder
  python scripts/collect.py --source /mnt/data/rakuten.zip      # local zip
  python scripts/collect.py --from-ens                          # download from ENS
  RAKUTEN_RAW_SOURCE=/mnt/data/rakuten.zip python scripts/collect.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from rakuten_img import config


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_present() -> bool:
    """True if the canonical raw layout already looks complete."""
    if not (config.X_TRAIN_CSV.exists() and config.Y_TRAIN_CSV.exists()):
        return False
    if not config.RAW_IMAGES_DIR.exists():
        return False
    return any(config.RAW_IMAGES_DIR.glob("*.jpg"))


def _download(url: str, dest: Path) -> None:
    import requests

    print(f"⬇️  Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"✅ Downloaded -> {dest}")


def _find_one(root: Path, patterns) -> Path | None:
    for pat in patterns:
        hits = list(root.rglob(pat))
        if hits:
            return hits[0]
    return None


def _normalize_into_raw(src_root: Path) -> None:
    """Locate the CSVs and image_train dir anywhere under src_root and place
    them into the canonical data/raw/ layout."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / "images").mkdir(parents=True, exist_ok=True)

    x_csv = _find_one(src_root, ["X_train_update.csv", "X_train*.csv"])
    y_csv = _find_one(src_root, ["Y_train_CVw08PX.csv", "Y_train*.csv"])
    img_dir = _find_one(src_root, ["image_train"])
    if img_dir is None or not img_dir.is_dir():
        # maybe images sit loose in a folder; fall back to first dir with jpgs
        for d in src_root.rglob("*"):
            if d.is_dir() and any(d.glob("*.jpg")):
                img_dir = d
                break

    if not (x_csv and y_csv and img_dir):
        raise FileNotFoundError(
            "Could not locate X_train CSV, Y_train CSV and an image_train folder "
            f"under {src_root}. Found: x={x_csv}, y={y_csv}, images={img_dir}"
        )

    shutil.copy2(x_csv, config.X_TRAIN_CSV)
    shutil.copy2(y_csv, config.Y_TRAIN_CSV)

    target_img = config.RAW_IMAGES_DIR
    if target_img.exists():
        shutil.rmtree(target_img)
    shutil.copytree(img_dir, target_img)
    print(f"✅ Normalized into {config.RAW_DIR}")


def acquire_from_ens() -> dict:
    """Authenticated download from ENS, then extract/normalize into data/raw/."""
    from rakuten_img import ens_download

    meta = {"source": f"ens:challenge/{config.ENS_CHALLENGE_ID}"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        downloaded = ens_download.download_all(
            config.ENS_USERNAME, config.ENS_PASSWORD, tmp_root / "downloads"
        )
        meta["files"] = [p.name for p in downloaded]

        # Extract any archives (the image archive is a zip), leave CSVs as-is.
        extract_root = tmp_root / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        for p in downloaded:
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as z:
                    z.extractall(extract_root)
            else:
                shutil.copy2(p, extract_root / p.name)

        _normalize_into_raw(extract_root)
    return meta


def acquire(source: str) -> dict:
    """Resolve the source, land+extract data into data/raw/, return source meta."""
    meta = {"source": source}
    src_path = Path(source)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        if source.startswith("http://") or source.startswith("https://"):
            zip_path = tmp_root / "download.zip"
            _download(source, zip_path)
            meta["sha256"] = _sha256(zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp_root / "extracted")
            _normalize_into_raw(tmp_root / "extracted")

        elif src_path.is_file() and src_path.suffix == ".zip":
            meta["sha256"] = _sha256(src_path)
            with zipfile.ZipFile(src_path) as z:
                z.extractall(tmp_root / "extracted")
            _normalize_into_raw(tmp_root / "extracted")

        elif src_path.is_dir():
            _normalize_into_raw(src_path)

        else:
            raise ValueError(
                f"Source {source!r} is not a .zip file, a directory, or an http(s) URL."
            )
    return meta


def verify() -> dict:
    """Sanity-check the acquired data; raise on failure, return counts."""
    x = pd.read_csv(config.X_TRAIN_CSV, index_col=0)
    y = pd.read_csv(config.Y_TRAIN_CSV, index_col=0)
    n_images = sum(1 for _ in config.RAW_IMAGES_DIR.glob("*.jpg"))

    problems = []
    if len(x) != len(y):
        problems.append(f"row mismatch: X={len(x)} vs Y={len(y)}")
    if n_images == 0:
        problems.append("no .jpg images found")
    if problems:
        raise RuntimeError("Verification failed: " + "; ".join(problems))

    counts = {"x_rows": int(len(x)), "y_rows": int(len(y)), "images": int(n_images)}
    print(f"✅ Verified: {counts}")
    return counts


def write_manifest(source_meta: dict, counts: dict) -> dict:
    manifest = {
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        **source_meta,
        "counts": counts,
        "canonical_paths": {
            "x_train": str(config.X_TRAIN_CSV),
            "y_train": str(config.Y_TRAIN_CSV),
            "images": str(config.RAW_IMAGES_DIR),
        },
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"📝 Wrote manifest -> {config.MANIFEST_PATH}")
    return manifest


def main() -> dict:
    # Load a local .env (ENS_USERNAME / ENS_PASSWORD etc.) if python-dotenv is present.
    try:
        from dotenv import load_dotenv
        load_dotenv(config.PROJECT_ROOT / ".env")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Acquire raw Rakuten data into data/raw/.")
    ap.add_argument("--source", default=config.RAW_SOURCE,
                    help="Local .zip, local dir, or http(s) URL. "
                         "Defaults to $RAKUTEN_RAW_SOURCE.")
    ap.add_argument("--from-ens", action="store_true",
                    help="Download from challengedata.ens.fr using "
                         "ENS_USERNAME / ENS_PASSWORD (env or .env).")
    ap.add_argument("--force", action="store_true",
                    help="Re-acquire even if data already present.")
    args = ap.parse_args()

    if is_present() and not args.force:
        print("✅ Raw data already present — skipping (use --force to re-acquire).")
        counts = verify()
        if not config.MANIFEST_PATH.exists():
            return write_manifest({"source": "pre-existing"}, counts)
        return json.loads(config.MANIFEST_PATH.read_text())

    if args.from_ens:
        source_meta = acquire_from_ens()
    elif args.source:
        source_meta = acquire(args.source)
    else:
        ap.error("No source. Pass --from-ens, or --source, or set RAKUTEN_RAW_SOURCE.")

    counts = verify()
    manifest = write_manifest(source_meta, counts)
    print("🎉 collect.py done.")
    return manifest


if __name__ == "__main__":
    main()