#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from lerobot.utils.gamiphy import (
    DEFAULT_BASE_URL,
    GamiphyAuthError,
    GamiphyClient,
    GamiphyNetworkError,
    GamiphyUploadError,
    GamiphyValidationError,
    _clean_metadata,
    _validate_base_url,
    build_file_manifest,
    collect_dataset_files,
    get_gamiphy_api_key,
    upload_dataset_to_gamiphy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal synthetic dataset directory structure."""
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"parquet")

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"fps": 30}')
    (tmp_path / "meta" / "stats.json").write_text("{}")
    (tmp_path / "meta" / "tasks.parquet").write_bytes(b"parquet")

    (tmp_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"parquet")

    (tmp_path / "videos" / "obs.images.top" / "chunk-000").mkdir(parents=True)
    (tmp_path / "videos" / "obs.images.top" / "chunk-000" / "file-000.mp4").write_bytes(b"mp4data")

    # images/ should be excluded
    (tmp_path / "images" / "obs.images.top").mkdir(parents=True)
    (tmp_path / "images" / "obs.images.top" / "frame-000.png").write_bytes(b"png")

    return tmp_path


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("GAMIPHY_UPLOAD_KEY", "test-key-abc123")
    return "test-key-abc123"


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("GAMIPHY_UPLOAD_KEY", raising=False)
    with pytest.raises(GamiphyAuthError, match="GAMIPHY_UPLOAD_KEY"):
        get_gamiphy_api_key()


def test_get_api_key_empty(monkeypatch):
    monkeypatch.setenv("GAMIPHY_UPLOAD_KEY", "   ")
    with pytest.raises(GamiphyAuthError, match="GAMIPHY_UPLOAD_KEY"):
        get_gamiphy_api_key()


def test_get_api_key_present(monkeypatch):
    monkeypatch.setenv("GAMIPHY_UPLOAD_KEY", "my-secret-key")
    assert get_gamiphy_api_key() == "my-secret-key"


# ---------------------------------------------------------------------------
# URL validation tests
# ---------------------------------------------------------------------------


def test_validate_base_url_https_allowed():
    assert _validate_base_url("https://api.gamiphy.ai") == "https://api.gamiphy.ai"


def test_validate_base_url_trailing_slash_stripped():
    assert _validate_base_url("https://api.gamiphy.ai/") == "https://api.gamiphy.ai"


def test_validate_base_url_localhost_http_allowed():
    assert _validate_base_url("http://localhost:8000") == "http://localhost:8000"


def test_validate_base_url_127_http_allowed():
    assert _validate_base_url("http://127.0.0.1:9000") == "http://127.0.0.1:9000"


def test_validate_base_url_http_non_local_raises():
    with pytest.raises(GamiphyValidationError, match="HTTPS"):
        _validate_base_url("http://api.gamiphy.ai")


# ---------------------------------------------------------------------------
# File collection tests
# ---------------------------------------------------------------------------


def test_collect_dataset_files_excludes_images(tmp_dataset):
    files = collect_dataset_files(tmp_dataset)
    rel_paths = [f.relative_to(tmp_dataset).as_posix() for f in files]
    # The top-level images/ directory must be excluded entirely.
    assert not any(p.startswith("images/") for p in rel_paths)


def test_collect_dataset_files_includes_expected(tmp_dataset):
    files = collect_dataset_files(tmp_dataset)
    rel_paths = {f.relative_to(tmp_dataset).as_posix() for f in files}
    assert "data/chunk-000/file-000.parquet" in rel_paths
    assert "meta/info.json" in rel_paths
    assert "meta/tasks.parquet" in rel_paths
    assert "meta/episodes/chunk-000/file-000.parquet" in rel_paths
    assert "videos/obs.images.top/chunk-000/file-000.mp4" in rel_paths


def test_collect_dataset_files_stable_order(tmp_dataset):
    files1 = collect_dataset_files(tmp_dataset)
    files2 = collect_dataset_files(tmp_dataset)
    assert files1 == files2


def test_collect_dataset_files_no_videos(tmp_path):
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"x")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    files = collect_dataset_files(tmp_path)
    assert any(f.suffix == ".parquet" for f in files)
    assert not any(f.suffix == ".mp4" for f in files)


# ---------------------------------------------------------------------------
# File manifest tests
# ---------------------------------------------------------------------------


def test_build_file_manifest_posix_paths(tmp_dataset):
    files = collect_dataset_files(tmp_dataset)
    manifest = build_file_manifest(tmp_dataset, files)
    for entry in manifest:
        assert "\\" not in entry["path"], "Manifest paths must be POSIX (no backslashes)"


def test_build_file_manifest_content_types(tmp_dataset):
    files = collect_dataset_files(tmp_dataset)
    manifest = build_file_manifest(tmp_dataset, files)
    by_path = {e["path"]: e for e in manifest}
    assert by_path["meta/info.json"]["content_type"] == "application/json"
    assert by_path["data/chunk-000/file-000.parquet"]["content_type"] == "application/octet-stream"
    assert by_path["videos/obs.images.top/chunk-000/file-000.mp4"]["content_type"] == "video/mp4"


def test_build_file_manifest_size_bytes(tmp_dataset):
    files = collect_dataset_files(tmp_dataset)
    manifest = build_file_manifest(tmp_dataset, files)
    for entry in manifest:
        assert isinstance(entry["size_bytes"], int)
        assert entry["size_bytes"] >= 0


# ---------------------------------------------------------------------------
# GamiphyClient — create_dataset
# ---------------------------------------------------------------------------


def _make_client(api_key="test-key", base_url="https://api.gamiphy.ai"):
    return GamiphyClient(api_key=api_key, base_url=base_url)


def _mock_response(status_code: int, json_data: dict | None = None, text: str = ""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_create_dataset_success():
    client = _make_client()
    mock_resp = _mock_response(
        200,
        {
            "dataset_id": "ds-123",
            "upload_instructions": {
                "data/chunk-000/file-000.parquet": {"url": "https://s3.example.com/signed", "method": "PUT", "headers": {}},
            },
        },
    )
    with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
        dataset_id, instructions = client.create_dataset("mybot", "alice/mybot", {}, [])
    assert dataset_id == "ds-123"
    assert "data/chunk-000/file-000.parquet" in instructions
    mock_post.assert_called_once()
    body = mock_post.call_args.kwargs["json"]
    assert body["display_name"] == "mybot"
    assert body["source_repo_id"] == "alice/mybot"


def test_create_dataset_401_raises_auth_error():
    client = _make_client()
    with patch.object(client._session, "post", return_value=_mock_response(401)):
        with pytest.raises(GamiphyAuthError, match="Authentication failed"):
            client.create_dataset("mybot", "alice/mybot", {}, [])


def test_create_dataset_422_raises_validation_error():
    client = _make_client()
    with patch.object(client._session, "post", return_value=_mock_response(422, text="bad request")):
        with pytest.raises(GamiphyValidationError):
            client.create_dataset("mybot", "alice/mybot", {}, [])


def test_create_dataset_connection_error_raises_network_error():
    client = _make_client()
    with patch.object(client._session, "post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(GamiphyNetworkError):
            client.create_dataset("mybot", "alice/mybot", {}, [])


def test_create_dataset_timeout_raises_network_error():
    client = _make_client()
    with patch.object(client._session, "post", side_effect=requests.Timeout()):
        with pytest.raises(GamiphyNetworkError):
            client.create_dataset("mybot", "alice/mybot", {}, [])


# ---------------------------------------------------------------------------
# GamiphyClient — upload_file
# ---------------------------------------------------------------------------


def test_upload_file_success(tmp_path):
    client = _make_client()
    f = tmp_path / "file.parquet"
    f.write_bytes(b"data")
    instruction = {"url": "https://s3.example.com/signed", "method": "PUT", "headers": {"Content-Type": "application/octet-stream"}}
    mock_resp = _mock_response(200)
    with patch("lerobot.utils.gamiphy.requests.request", return_value=mock_resp) as mock_req:
        client.upload_file(instruction, f)
    mock_req.assert_called_once()
    call_kwargs = mock_req.call_args.kwargs
    assert call_kwargs["method"] == "PUT"
    assert call_kwargs["url"] == instruction["url"]
    assert call_kwargs["headers"] == instruction["headers"]


def test_upload_file_no_bearer_header(tmp_path):
    """The signed URL PUT must NOT include the Authorization header."""
    client = _make_client()
    f = tmp_path / "file.parquet"
    f.write_bytes(b"data")
    instruction = {"url": "https://s3.example.com/signed", "method": "PUT", "headers": {}}
    mock_resp = _mock_response(200)
    with patch("lerobot.utils.gamiphy.requests.request", return_value=mock_resp) as mock_req:
        client.upload_file(instruction, f)
    headers_sent = mock_req.call_args.kwargs.get("headers", {})
    assert "Authorization" not in headers_sent


def test_upload_file_http_error_raises(tmp_path):
    client = _make_client()
    f = tmp_path / "file.parquet"
    f.write_bytes(b"data")
    instruction = {"url": "https://s3.example.com/signed", "method": "PUT", "headers": {}}
    with patch("lerobot.utils.gamiphy.requests.request", return_value=_mock_response(403)):
        with pytest.raises(GamiphyNetworkError, match="Failed to upload"):
            client.upload_file(instruction, f)


def test_upload_file_connection_error_raises(tmp_path):
    client = _make_client()
    f = tmp_path / "file.parquet"
    f.write_bytes(b"data")
    instruction = {"url": "https://s3.example.com/signed", "method": "PUT", "headers": {}}
    with patch("lerobot.utils.gamiphy.requests.request", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(GamiphyNetworkError, match="Network error"):
            client.upload_file(instruction, f)


# ---------------------------------------------------------------------------
# GamiphyClient — confirm_dataset
# ---------------------------------------------------------------------------


def test_confirm_dataset_success():
    client = _make_client()
    with patch.object(client._session, "post", return_value=_mock_response(200)):
        client.confirm_dataset("ds-123")  # should not raise


def test_confirm_dataset_error_raises():
    client = _make_client()
    with patch.object(client._session, "post", return_value=_mock_response(500, text="server error")):
        with pytest.raises(GamiphyUploadError):
            client.confirm_dataset("ds-123")


# ---------------------------------------------------------------------------
# GamiphyClient — abort_dataset
# ---------------------------------------------------------------------------


def test_abort_dataset_failure_logs_warning_not_raises(caplog):
    client = _make_client()
    with patch.object(client._session, "post", return_value=_mock_response(500)):
        with caplog.at_level("WARNING"):
            client.abort_dataset("ds-999")  # must not raise
    assert any("abort" in r.message.lower() or "ds-999" in r.message for r in caplog.records)


def test_abort_dataset_connection_error_logs_not_raises(caplog):
    client = _make_client()
    with patch.object(client._session, "post", side_effect=Exception("boom")):
        with caplog.at_level("WARNING"):
            client.abort_dataset("ds-999")  # must not raise


# ---------------------------------------------------------------------------
# upload_dataset_to_gamiphy — full pipeline
# ---------------------------------------------------------------------------


def _make_upload_instructions(root: Path, files: list[Path]) -> dict:
    return {
        f.relative_to(root).as_posix(): {
            "url": f"https://s3.example.com/{f.name}",
            "method": "PUT",
            "headers": {},
        }
        for f in files
    }


def test_upload_dataset_full_pipeline(tmp_dataset, api_key):
    from lerobot.utils.gamiphy import collect_dataset_files

    files = collect_dataset_files(tmp_dataset)
    instructions = _make_upload_instructions(tmp_dataset, files)

    meta_info = {
        "fps": 30,
        "total_episodes": 5,
        "total_frames": 150,
        "total_tasks": 1,
        "features": {},
        "robot_type": "so100",
    }

    with (
        patch("lerobot.utils.gamiphy.GamiphyClient.create_dataset") as mock_create,
        patch("lerobot.utils.gamiphy.GamiphyClient.upload_file") as mock_upload,
        patch("lerobot.utils.gamiphy.GamiphyClient.confirm_dataset") as mock_confirm,
        patch("lerobot.utils.gamiphy.GamiphyClient.abort_dataset") as mock_abort,
    ):
        mock_create.return_value = ("ds-abc", instructions)

        result = upload_dataset_to_gamiphy(
            root=tmp_dataset,
            repo_id="alice/mybot",
            meta_info=meta_info,
        )

    assert result == "ds-abc"
    mock_create.assert_called_once()
    create_kwargs = mock_create.call_args.kwargs
    assert create_kwargs["display_name"] == "mybot"
    assert create_kwargs["source_repo_id"] == "alice/mybot"
    assert mock_upload.call_count == len(files)
    mock_confirm.assert_called_once_with("ds-abc")
    mock_abort.assert_not_called()


def test_upload_dataset_abort_on_upload_failure(tmp_dataset, api_key):
    from lerobot.utils.gamiphy import collect_dataset_files

    files = collect_dataset_files(tmp_dataset)
    instructions = _make_upload_instructions(tmp_dataset, files)

    with (
        patch("lerobot.utils.gamiphy.GamiphyClient.create_dataset") as mock_create,
        patch("lerobot.utils.gamiphy.GamiphyClient.upload_file") as mock_upload,
        patch("lerobot.utils.gamiphy.GamiphyClient.confirm_dataset") as mock_confirm,
        patch("lerobot.utils.gamiphy.GamiphyClient.abort_dataset") as mock_abort,
    ):
        mock_create.return_value = ("ds-abc", instructions)
        mock_upload.side_effect = GamiphyNetworkError("upload failed")

        with pytest.raises(GamiphyNetworkError):
            upload_dataset_to_gamiphy(
                root=tmp_dataset,
                repo_id="alice/mybot",
                meta_info={},
            )

    mock_abort.assert_called_once_with("ds-abc")
    mock_confirm.assert_not_called()


def test_upload_dataset_missing_url_aborts(tmp_dataset, api_key):
    with (
        patch("lerobot.utils.gamiphy.GamiphyClient.create_dataset") as mock_create,
        patch("lerobot.utils.gamiphy.GamiphyClient.confirm_dataset") as mock_confirm,
        patch("lerobot.utils.gamiphy.GamiphyClient.abort_dataset") as mock_abort,
    ):
        # Return empty instructions — server didn't provide URLs for any file
        mock_create.return_value = ("ds-abc", {})

        with pytest.raises(GamiphyUploadError, match="upload URL"):
            upload_dataset_to_gamiphy(
                root=tmp_dataset,
                repo_id="alice/mybot",
                meta_info={},
            )

    mock_abort.assert_called_once()
    mock_confirm.assert_not_called()


def test_upload_dataset_no_api_key_fast_fails(tmp_dataset, monkeypatch):
    monkeypatch.delenv("GAMIPHY_UPLOAD_KEY", raising=False)
    with (
        patch("lerobot.utils.gamiphy.GamiphyClient.create_dataset") as mock_create,
    ):
        with pytest.raises(GamiphyAuthError):
            upload_dataset_to_gamiphy(
                root=tmp_dataset,
                repo_id="alice/mybot",
                meta_info={},
            )
    mock_create.assert_not_called()


def test_upload_dataset_empty_directory_raises(tmp_path, api_key):
    with pytest.raises(GamiphyUploadError, match="No dataset files found"):
        upload_dataset_to_gamiphy(root=tmp_path, repo_id="alice/mybot", meta_info={})


# ---------------------------------------------------------------------------
# _clean_metadata
# ---------------------------------------------------------------------------


def test_clean_metadata_filters_fields():
    raw = {
        "fps": 30,
        "total_episodes": 10,
        "total_frames": 300,
        "total_tasks": 2,
        "features": {"obs": {}},
        "robot_type": "so100",
        "codebase_version": "v3.0",       # internal — should be excluded
        "chunks_size": 1000,              # internal — should be excluded
    }
    clean = _clean_metadata(raw)
    assert "codebase_version" not in clean
    assert "chunks_size" not in clean
    assert clean["fps"] == 30
    assert clean["robot_type"] == "so100"
