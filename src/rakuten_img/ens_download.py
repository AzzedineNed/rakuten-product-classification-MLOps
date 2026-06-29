"""Authenticated download from the ENS Challenge Data platform.

challengedata.ens.fr is a Django site: an unauthenticated request to a download
URL 302-redirects to /login/?next=... . So we:
  1. GET the login page and scrape its csrfmiddlewaretoken hidden field.
  2. POST username + password (+ that token) with a Referer header; Django sets
     a 'sessionid' cookie on success.
  3. Stream each download endpoint to disk with the authenticated session.

Credentials are read from the environment (ENS_USERNAME / ENS_PASSWORD) and are
NEVER logged or persisted. This module does not store secrets anywhere.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config

_CSRF_RE = re.compile(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"')
_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.IGNORECASE)
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _extract_csrf(html: str) -> str:
    m = _CSRF_RE.search(html)
    if not m:
        raise RuntimeError("Could not find csrfmiddlewaretoken on the login page "
                           "(site layout may have changed).")
    return m.group(1)


def _filename_from_response(resp, fallback: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = _FILENAME_RE.search(cd)
    return (m.group(1).strip() if m else fallback)


def login(username: str, password: str):
    """Return an authenticated requests.Session, or raise on failure."""
    import requests

    if not username or not password:
        raise ValueError(
            "Missing ENS credentials. Set ENS_USERNAME and ENS_PASSWORD "
            "(e.g. in a .env file) before using --from-ens."
        )

    session = requests.Session()
    session.headers.update({"User-Agent": _UA})
    login_url = f"{config.ENS_BASE}/login/"

    page = session.get(login_url, timeout=30)
    page.raise_for_status()
    token = _extract_csrf(page.text)

    resp = session.post(
        login_url,
        data={
            "csrfmiddlewaretoken": token,
            "username": username,
            "password": password,
            "next": "",
        },
        headers={"Referer": login_url},
        timeout=30,
    )
    resp.raise_for_status()

    # Django issues a 'sessionid' cookie only on a successful login.
    if "sessionid" not in session.cookies.get_dict():
        raise RuntimeError(
            "Login failed — check ENS_USERNAME / ENS_PASSWORD. "
            "(No session cookie was issued.)"
        )
    print("🔐 Logged in to ENS Challenge Data.")
    return session


def download_slug(session, slug: str, dest_dir: Path) -> Path:
    """Stream one challenge download endpoint to dest_dir. Returns the file path."""
    from tqdm.auto import tqdm

    url = f"{config.ENS_BASE}/participants/challenges/{config.ENS_CHALLENGE_ID}/download/{slug}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        # If we got HTML back, we were bounced to the login page = not authed.
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"Expected a file for '{slug}' but got HTML "
                               f"(session not authenticated?).")
        fname = _filename_from_response(r, fallback=f"{slug}.bin")
        out_path = dest_dir / fname
        total = int(r.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download:{slug}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"✅ {slug} -> {out_path.name}")
    return out_path


def download_all(username: str, password: str, dest_dir: Path) -> list[Path]:
    """Log in and download the train CSVs + image archive into dest_dir."""
    session = login(username, password)
    paths = []
    for slug in config.ENS_DOWNLOAD_SLUGS:
        paths.append(download_slug(session, slug, dest_dir))
    return paths
