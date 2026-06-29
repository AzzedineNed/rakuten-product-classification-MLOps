"""Tests for the ENS downloader's parsing helpers. No network access — we only
exercise the pure string parsing (CSRF token + Content-Disposition filename).
"""
import pytest

from rakuten_img import ens_download


def test_extract_csrf_token():
    html = '''
    <form method="post" action="/login/">
      <input type="hidden" name="csrfmiddlewaretoken" value="AbC123xyzTOKEN">
      <input name="username"><input name="password" type="password">
    </form>
    '''
    assert ens_download._extract_csrf(html) == "AbC123xyzTOKEN"


def test_extract_csrf_missing_raises():
    with pytest.raises(RuntimeError):
        ens_download._extract_csrf("<form>no token here</form>")


class _FakeResp:
    def __init__(self, cd):
        self.headers = {"Content-Disposition": cd}


def test_filename_from_content_disposition():
    r = _FakeResp('attachment; filename="X_train_update.csv"')
    assert ens_download._filename_from_response(r, "fallback.bin") == "X_train_update.csv"


def test_filename_fallback_when_absent():
    r = _FakeResp("")
    assert ens_download._filename_from_response(r, "supplementary-files.bin") == \
        "supplementary-files.bin"
