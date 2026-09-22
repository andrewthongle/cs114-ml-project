"""Synthetic ZIP/network fixtures only; never contact or download ViHSD."""
import hashlib
import io
import json
from urllib.error import HTTPError, URLError
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from safeview_ml import cli, remote_data
from safeview_ml.data import load_local_dataset

SHA = "a" * 40
CSV = "free_text,label_id\n  xin chào ,0\nví dụ một,1\nví dụ hai,2\n"


def _zip_payload(changes=None):
    files = {f"vihsd/{name}.csv": CSV for name in ("train", "dev", "test")}
    files.update(changes or {})
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            if content is not None:
                archive.writestr(name, content)
    return buffer.getvalue()


def _mock_github(monkeypatch, payload=None):
    calls = []
    payload = _zip_payload() if payload is None else payload

    def fetch(url, maximum):
        calls.append(url)
        return json.dumps({"sha": SHA}).encode() if "api.github.com" in url else payload

    monkeypatch.setattr(remote_data, "_fetch_bytes", fetch)
    return calls


def test_github_is_pinned_in_memory_and_matches_local_fingerprint(tmp_path, monkeypatch):
    payload = _zip_payload({"vihsd/test.csv": "free_text,label_id\nNA,0\n123,2\n"})
    calls = _mock_github(monkeypatch, payload)
    monkeypatch.chdir(tmp_path)
    bundle = remote_data.load_github_dataset()
    assert list(tmp_path.iterdir()) == []
    assert calls == [
        "https://api.github.com/repos/sonlam1102/vihsd/commits/main",
        f"https://raw.githubusercontent.com/sonlam1102/vihsd/{SHA}/data/vihsd.zip",
    ]
    assert bundle.manifest["revision"] == SHA
    assert bundle.manifest["requested_revision"] == "main"
    assert bundle.manifest["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert bundle.manifest["split_counts"] == {"train": 3, "validation": 3, "test": 2}
    assert bundle.splits["train"].text.iloc[0] == "  xin chào "
    assert bundle.splits["train"].label.tolist() == [0, 1, 2]
    assert bundle.splits["validation"].sample_id.iloc[0] == "validation:00000000"
    assert bundle.splits["test"].text.tolist() == ["NA", "123"]
    with ZipFile(io.BytesIO(payload)) as archive:
        for name in ("train", "dev", "test"):
            (tmp_path / f"{name}.csv").write_bytes(archive.read(f"vihsd/{name}.csv"))
    assert load_local_dataset(tmp_path).manifest["fingerprint"] == bundle.manifest["fingerprint"]


def test_explicit_sha_skips_api_and_optional_cache_reuses_zip(tmp_path, monkeypatch):
    calls = _mock_github(monkeypatch)
    cache = tmp_path / "runtime-cache"
    first = remote_data.load_github_dataset(SHA.upper(), cache_dir=cache)
    second = remote_data.load_github_dataset(SHA, cache_dir=cache)
    assert len(calls) == 1
    assert "api.github.com" not in calls[0]
    assert [path.name for path in cache.iterdir()] == [f"{SHA}.zip"]
    assert first.manifest["fingerprint"] == second.manifest["fingerprint"]


@pytest.mark.parametrize("payload, message", [
    (b"not a ZIP", "archive is invalid"),
    (_zip_payload({"vihsd/dev.csv": None}), "missing vihsd/dev.csv"),
    (_zip_payload({"vihsd/train.csv": "free_text,label_id\nsynthetic,9\n"}), "Invalid label"),
    (_zip_payload({"../../escaped.csv": CSV}), "unsafe member paths"),
    (_zip_payload({"vihsd/train.csv": "other,label_id\nvalue,0\n"}), "expected a text column"),
])
def test_invalid_archives_fail_without_cache_or_substitute(tmp_path, monkeypatch, payload, message):
    _mock_github(monkeypatch, payload)
    cache = tmp_path / "runtime-cache"
    with pytest.raises(ValueError, match=message):
        remote_data.load_github_dataset(SHA, cache_dir=cache)
    assert not cache.exists()


def test_uncompressed_limit_is_checked_before_reading_members(monkeypatch):
    _mock_github(monkeypatch)
    monkeypatch.setattr(remote_data, "MAX_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(ValueError, match="uncompressed size limit"):
        remote_data.load_github_dataset(SHA)


@pytest.mark.parametrize("response", [{"sha": "main"}, [], {"sha": None}])
def test_invalid_revision_metadata_fails_without_fetching_archive(monkeypatch, response):
    calls = []

    def fetch(url, maximum):
        calls.append(url)
        return json.dumps(response).encode()

    monkeypatch.setattr(remote_data, "_fetch_bytes", fetch)
    with pytest.raises(remote_data.DatasetAccessError, match="immutable commit SHA"):
        remote_data.load_github_dataset()
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    HTTPError("https://api.github.com", 403, "Forbidden", {}, None),
    URLError("offline"),
    TimeoutError("timed out"),
])
def test_network_failure_is_clear_and_does_not_write(tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(remote_data, "urlopen", fail)
    with pytest.raises(remote_data.DatasetAccessError, match="[Nn]o replacement dataset"):
        remote_data.load_github_dataset(cache_dir=tmp_path / "cache")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("content_length", [None, "100"])
def test_network_size_limit_and_timeout(monkeypatch, content_length):
    response = io.BytesIO(b"0123456789")
    response.headers = {"Content-Length": content_length} if content_length else {}
    calls = []

    def open_response(request, timeout):
        calls.append((request.full_url, timeout))
        return response

    monkeypatch.setattr(remote_data, "urlopen", open_response)
    with pytest.raises(ValueError, match="size limit"):
        remote_data._fetch_bytes("https://raw.githubusercontent.com/test", 5)
    assert calls[0][1] == remote_data.REQUEST_TIMEOUT_SECONDS


@pytest.mark.parametrize("explicit_local", [False, True])
def test_cli_defaults_to_github_but_preserves_explicit_local_directory(tmp_path, monkeypatch, explicit_local):
    _mock_github(monkeypatch)
    bundle = remote_data.load_github_dataset(SHA)
    calls = []
    monkeypatch.setattr(cli, "load_github_dataset", lambda **kwargs: calls.append(("github", kwargs)) or bundle)
    monkeypatch.setattr(cli, "load_local_dataset", lambda directory, **kwargs: calls.append(("local", directory)) or bundle)
    monkeypatch.setattr(cli, "export_eda", lambda *args: {})
    args = ["eda", "--project-dir", str(tmp_path)]
    if explicit_local:
        args += ["--data-dir", "/synthetic/local"]
    assert cli.main(args) == 0
    assert calls == ([("local", "/synthetic/local")] if explicit_local else [("github", {"revision": "main", "cache_dir": None})])
    assert json.loads((tmp_path / "data/manifest.json").read_text())["revision"] == SHA


def test_cli_rejects_conflicting_source_without_network(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_github_dataset", lambda **kwargs: pytest.fail("Must not fetch on conflicting options"))
    assert cli.main(["eda", "--data-source", "github", "--data-dir", "local"]) == 2
    assert "choose --data-source local" in capsys.readouterr().err
