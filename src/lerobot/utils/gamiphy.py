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

"""Gamiphy.ai dataset upload client.

Uploads lerobot datasets to gamiphy.ai via a pre-signed URL workflow:
  1. POST /api/datasets with name, metadata, and file manifest → signed upload instructions
  2. PUT each file to its signed URL (streaming, with provider-specific headers)
  3. POST /api/datasets/{id}/confirm on success, or /abort on failure
"""

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lerobot.datasets.utils import DATA_DIR, EPISODES_DIR, VIDEO_DIR

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.gamiphy.ai"
DEFAULT_CONNECT_TIMEOUT = 10.0   # seconds
DEFAULT_READ_TIMEOUT = 300.0     # seconds for large file uploads
DEFAULT_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GamiphyUploadError(RuntimeError):
    """Base exception for all gamiphy.ai upload failures."""


class GamiphyAuthError(GamiphyUploadError):
    """Raised on authentication/authorization failures (HTTP 401/403)."""


class GamiphyValidationError(GamiphyUploadError):
    """Raised on request validation failures (HTTP 400/422)."""


class GamiphyNetworkError(GamiphyUploadError):
    """Raised on network-level failures (connection errors, timeouts)."""


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def get_gamiphy_api_key() -> str:
    """Read GAMIPHY_UPLOAD_KEY from the environment.

    Raises:
        GamiphyAuthError: If the variable is absent or empty.
    """
    key = os.environ.get("GAMIPHY_UPLOAD_KEY", "").strip()
    if not key:
        raise GamiphyAuthError(
            "GAMIPHY_UPLOAD_KEY environment variable is not set or empty. "
            "Set it to your gamiphy.ai API key before uploading. "
            "You can generate an upload key in your gamiphy.ai dashboard."
        )
    return key


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


def _validate_base_url(url: str) -> str:
    """Validate and normalise the API base URL.

    HTTPS is required for non-localhost hosts to prevent token exfiltration.
    http://localhost and http://127.0.0.1 are allowed for local development.

    Raises:
        GamiphyValidationError: If the URL uses HTTP with a non-local host.
    """
    parsed = urlparse(url.rstrip("/"))
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if parsed.scheme == "http" and not is_local:
        raise GamiphyValidationError(
            f"Insecure base URL '{url}'. gamiphy_base_url must use HTTPS for non-localhost hosts."
        )
    return url.rstrip("/")


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_dataset_files(root: Path) -> list[Path]:
    """Return all dataset files to upload from *root*, in a stable order.

    Includes: data parquet files, metadata JSON/parquet, episode parquet files,
    and video MP4 files. Excludes the temporary ``images/`` directory.

    Uses path constants from :mod:`lerobot.datasets.utils` to stay in sync with
    the dataset format as it evolves.
    """
    patterns = [
        f"{DATA_DIR}/**/*.parquet",
        "meta/*.json",
        "meta/*.parquet",
        f"{EPISODES_DIR}/**/*.parquet",
        f"{VIDEO_DIR}/**/*.mp4",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    return files


# ---------------------------------------------------------------------------
# File manifest
# ---------------------------------------------------------------------------

_CONTENT_TYPES = {
    ".parquet": "application/octet-stream",
    ".json": "application/json",
    ".mp4": "video/mp4",
}


def build_file_manifest(root: Path, files: list[Path]) -> list[dict]:
    """Build the upload manifest sent to the gamiphy API.

    Each entry contains the POSIX-normalised relative path, file size in bytes,
    and a content-type derived from the file extension.

    Args:
        root: Dataset root directory (``self.root``).
        files: Absolute file paths returned by :func:`collect_dataset_files`.

    Returns:
        List of dicts with keys ``path``, ``size_bytes``, ``content_type``.
    """
    manifest = []
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        content_type = _CONTENT_TYPES.get(file_path.suffix, "application/octet-stream")
        manifest.append(
            {
                "path": rel,
                "size_bytes": file_path.stat().st_size,
                "content_type": content_type,
            }
        )
    return manifest


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class GamiphyClient:
    """HTTP client for the gamiphy.ai dataset upload API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.base_url = _validate_base_url(base_url)
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout

        # Session for authenticated API calls (no redirects — prevents token leakage).
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        self._session.max_redirects = 0

        retry = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _api_timeout(self) -> tuple[float, float]:
        return (self._connect_timeout, self._read_timeout)

    # ------------------------------------------------------------------

    def create_dataset(
        self,
        display_name: str,
        source_repo_id: str,
        metadata: dict,
        file_manifest: list[dict],
    ) -> tuple[str, dict[str, dict]]:
        """Register a new dataset and obtain per-file signed upload instructions.

        Args:
            display_name: Short name shown in the gamiphy dashboard.
            source_repo_id: Full lerobot ``repo_id`` (e.g. ``alice/pick-block``).
            metadata: Cleaned dataset metadata dict (fps, episodes, features, …).
            file_manifest: List returned by :func:`build_file_manifest`.

        Returns:
            ``(dataset_id, upload_instructions)`` where *upload_instructions* maps
            POSIX-relative paths to ``{url, method, headers}`` dicts.

        Raises:
            GamiphyAuthError: On HTTP 401/403.
            GamiphyValidationError: On HTTP 400/422.
            GamiphyNetworkError: On connection/timeout errors.
        """
        body = {
            "display_name": display_name,
            "source_repo_id": source_repo_id,
            "metadata": metadata,
            "files": file_manifest,
        }
        try:
            response = self._session.post(
                f"{self.base_url}/api/datasets",
                json=body,
                timeout=self._api_timeout(),
                allow_redirects=False,
            )
            self._raise_for_status(response)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise GamiphyNetworkError(f"Network error contacting gamiphy.ai: {e}") from e

        data = response.json()
        dataset_id: str = data["dataset_id"]
        upload_instructions: dict[str, dict] = data["upload_instructions"]
        return dataset_id, upload_instructions

    # ------------------------------------------------------------------

    def upload_file(
        self,
        upload_instruction: dict,
        file_path: Path,
    ) -> None:
        """Upload a single file using the signed upload instruction.

        The bearer token is NOT sent on the storage PUT request; signed URLs
        are self-authenticated. A plain ``requests.Session`` (no auth headers)
        is used for each PUT to avoid leaking credentials.

        Args:
            upload_instruction: Dict with keys ``url``, ``method``, ``headers``.
            file_path: Absolute local path to the file to upload.

        Raises:
            GamiphyNetworkError: On HTTP error or connection failure.
        """
        url = upload_instruction["url"]
        method = upload_instruction.get("method", "PUT").upper()
        extra_headers = upload_instruction.get("headers", {})

        try:
            with open(file_path, "rb") as fh:
                response = requests.request(
                    method=method,
                    url=url,
                    data=fh,
                    headers=extra_headers,
                    timeout=(self._connect_timeout, self._read_timeout),
                )
            if not response.ok:
                raise GamiphyNetworkError(
                    f"Failed to upload {file_path.name} (HTTP {response.status_code})"
                )
        except (requests.ConnectionError, requests.Timeout) as e:
            raise GamiphyNetworkError(
                f"Network error uploading {file_path.name}: {e}"
            ) from e

    # ------------------------------------------------------------------

    def confirm_dataset(self, dataset_id: str) -> None:
        """Mark the dataset as fully uploaded and ready in the gamiphy dashboard.

        Raises:
            GamiphyUploadError: On HTTP error or connection failure.
        """
        try:
            response = self._session.post(
                f"{self.base_url}/api/datasets/{dataset_id}/confirm",
                timeout=self._api_timeout(),
                allow_redirects=False,
            )
            self._raise_for_status(response)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise GamiphyNetworkError(
                f"Network error confirming dataset {dataset_id}: {e}"
            ) from e

    # ------------------------------------------------------------------

    def abort_dataset(self, dataset_id: str) -> None:
        """Delete an incomplete dataset record on gamiphy.ai.

        Best-effort: logs a warning on failure rather than raising, so that the
        caller's original exception is not suppressed.
        """
        try:
            response = self._session.post(
                f"{self.base_url}/api/datasets/{dataset_id}/abort",
                timeout=self._api_timeout(),
                allow_redirects=False,
            )
            if not response.ok:
                logger.warning(
                    "Failed to abort dataset %s on gamiphy.ai (HTTP %d). "
                    "The incomplete record may need to be removed manually.",
                    dataset_id,
                    response.status_code,
                )
        except Exception:
            logger.warning(
                "Could not reach gamiphy.ai to abort dataset %s. "
                "The incomplete record may need to be removed manually.",
                dataset_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Convert HTTP error responses into typed gamiphy exceptions."""
        if response.ok:
            return
        status = response.status_code
        try:
            body = response.text[:300]
        except Exception:
            body = ""
        if status in (401, 403):
            raise GamiphyAuthError(
                f"Authentication failed (HTTP {status}). "
                "Check that GAMIPHY_UPLOAD_KEY is valid and has upload permissions."
            )
        if status in (400, 422):
            raise GamiphyValidationError(
                f"Request rejected by gamiphy.ai (HTTP {status}): {body}"
            )
        raise GamiphyUploadError(
            f"Unexpected response from gamiphy.ai (HTTP {status}): {body}"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _clean_metadata(meta_info: dict) -> dict:
    """Extract the subset of meta_info fields relevant to the gamiphy dashboard."""
    return {
        "fps": meta_info.get("fps"),
        "total_episodes": meta_info.get("total_episodes"),
        "total_frames": meta_info.get("total_frames"),
        "total_tasks": meta_info.get("total_tasks", 0),
        "features": meta_info.get("features", {}),
        "robot_type": meta_info.get("robot_type"),
    }


def upload_dataset_to_gamiphy(
    root: Path,
    repo_id: str,
    meta_info: dict,
    base_url: str | None = None,
) -> str:
    """Upload a lerobot dataset to gamiphy.ai.

    This is the top-level entry point called from
    :meth:`LeRobotDataset.push_to_gamiphy`.

    Args:
        root: Local dataset root directory (``LeRobotDataset.root``).
        repo_id: Dataset repo identifier (e.g. ``alice/pick-block``).
        meta_info: ``LeRobotDataset.meta.info`` dict.
        base_url: Override the default gamiphy.ai API URL. Must be HTTPS
            unless the host is ``localhost`` or ``127.0.0.1``.

    Returns:
        The ``dataset_id`` assigned by gamiphy.ai.

    Raises:
        GamiphyAuthError: If GAMIPHY_UPLOAD_KEY is missing/invalid.
        GamiphyValidationError: If the server rejects the request.
        GamiphyNetworkError: On connection or upload failure.
        GamiphyUploadError: On any other gamiphy-specific error.
    """
    api_key = get_gamiphy_api_key()
    client = GamiphyClient(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)

    # Derive display name: part after "/" or full repo_id if no slash.
    display_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    files = collect_dataset_files(root)
    if not files:
        raise GamiphyUploadError(
            f"No dataset files found in {root}. "
            "Make sure the dataset has been finalized (dataset.finalize()) before uploading."
        )

    manifest = build_file_manifest(root, files)
    metadata = _clean_metadata(meta_info)

    logger.info("Creating dataset '%s' on gamiphy.ai...", display_name)
    dataset_id, upload_instructions = client.create_dataset(
        display_name=display_name,
        source_repo_id=repo_id,
        metadata=metadata,
        file_manifest=manifest,
    )
    logger.info(
        "Dataset created with id=%s (%d files to upload)", dataset_id, len(files)
    )

    # Upload all files; abort and re-raise on any failure.
    try:
        for i, file_path in enumerate(files):
            posix_key = file_path.relative_to(root).as_posix()
            if posix_key not in upload_instructions:
                raise GamiphyUploadError(
                    f"Server did not provide an upload URL for '{posix_key}'. "
                    "This may indicate an API version mismatch."
                )
            instruction = upload_instructions[posix_key]
            logger.info("Uploading [%d/%d] %s", i + 1, len(files), posix_key)
            client.upload_file(instruction, file_path)
    except Exception:
        logger.warning("Upload failed — aborting dataset %s on gamiphy.ai", dataset_id)
        client.abort_dataset(dataset_id)
        raise

    client.confirm_dataset(dataset_id)
    logger.info(
        "Dataset successfully uploaded to gamiphy.ai (id=%s)", dataset_id
    )
    return dataset_id
