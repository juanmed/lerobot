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

"""
Upload one or more locally-recorded lerobot datasets to gamiphy.ai.

Use this script when you recorded datasets without the --dataset.push_to_gamiphy
flag and want to upload them later.

Authentication: set the GAMIPHY_UPLOAD_KEY environment variable to your API key
from the gamiphy.ai dashboard before running.

Examples:

Upload a single dataset by repo_id (looks in $HF_LEROBOT_HOME/<repo_id>):
```shell
lerobot-gamiphy-upload --repo_id alice/pick-block
```

Upload from an explicit local path:
```shell
lerobot-gamiphy-upload --repo_id alice/pick-block --root /path/to/pick-block
```

Upload all datasets found directly inside a folder:
```shell
lerobot-gamiphy-upload --scan_dir /path/to/datasets
```

Upload to a staging server instead of production:
```shell
lerobot-gamiphy-upload --repo_id alice/pick-block --base_url https://staging.api.gamiphy.ai
```

Dry-run (discover files without actually uploading):
```shell
lerobot-gamiphy-upload --repo_id alice/pick-block --dry_run
```
"""

import argparse
import logging
import sys
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.gamiphy import (
    GamiphyUploadError,
    build_file_manifest,
    collect_dataset_files,
    upload_dataset_to_gamiphy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INFO_JSON = "meta/info.json"


def _load_meta_info(root: Path) -> dict:
    """Load meta/info.json from a dataset root. Returns empty dict on failure."""
    import json

    info_path = root / INFO_JSON
    if not info_path.exists():
        return {}
    try:
        with open(info_path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read %s: %s", info_path, exc)
        return {}


def _is_dataset_root(path: Path) -> bool:
    """Return True if *path* looks like a finalized lerobot dataset root."""
    return (path / INFO_JSON).exists() and (path / "data").is_dir()


def _upload_one(repo_id: str, root: Path, base_url: str | None, dry_run: bool) -> bool:
    """Upload a single dataset. Returns True on success, False on failure."""
    if not _is_dataset_root(root):
        logger.error(
            "No dataset found at %s (missing meta/info.json or data/ directory). Skipping.", root
        )
        return False

    meta_info = _load_meta_info(root)
    files = collect_dataset_files(root)

    if not files:
        logger.error("No uploadable files found in %s. Skipping.", root)
        return False

    if dry_run:
        manifest = build_file_manifest(root, files)
        total_mb = sum(e["size_bytes"] for e in manifest) / (1024 * 1024)
        logger.info(
            "[DRY RUN] Would upload %d files (%.1f MB) from %s as '%s'",
            len(files),
            total_mb,
            root,
            repo_id,
        )
        for entry in manifest:
            logger.info("  %s  (%.1f KB)", entry["path"], entry["size_bytes"] / 1024)
        return True

    try:
        dataset_id = upload_dataset_to_gamiphy(
            root=root,
            repo_id=repo_id,
            meta_info=meta_info,
            base_url=base_url,
        )
        logger.info("Uploaded '%s' → gamiphy dataset id=%s", repo_id, dataset_id)
        return True
    except GamiphyUploadError as exc:
        logger.error("Upload failed for '%s': %s", repo_id, exc)
        return False


def main():
    parser = argparse.ArgumentParser(
        prog="lerobot-gamiphy-upload",
        description="Upload locally-recorded lerobot datasets to gamiphy.ai.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--repo_id",
        type=str,
        metavar="USER/DATASET",
        help=(
            "Dataset repo identifier (e.g. alice/pick-block). "
            "The dataset is loaded from $HF_LEROBOT_HOME/<repo_id> unless --root is provided."
        ),
    )
    target.add_argument(
        "--scan_dir",
        type=Path,
        metavar="DIR",
        help=(
            "Scan this directory for dataset sub-folders and upload each one. "
            "The repo_id for each dataset is inferred from its folder name as "
            "<parent_folder_name>/<sub_folder_name>."
        ),
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Explicit local path to the dataset root (only used with --repo_id). "
            "Defaults to $HF_LEROBOT_HOME/<repo_id>."
        ),
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        metavar="URL",
        help="Override the gamiphy.ai API base URL (e.g. https://staging.api.gamiphy.ai). "
        "Must use HTTPS unless the host is localhost or 127.0.0.1.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover and list files that would be uploaded without actually uploading.",
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("--- DRY RUN — no files will be uploaded ---")

    # ---- Single dataset upload ----
    if args.repo_id:
        root = args.root if args.root is not None else HF_LEROBOT_HOME / args.repo_id
        success = _upload_one(
            repo_id=args.repo_id,
            root=root,
            base_url=args.base_url,
            dry_run=args.dry_run,
        )
        sys.exit(0 if success else 1)

    # ---- Scan directory ----
    scan_dir: Path = args.scan_dir
    if not scan_dir.is_dir():
        logger.error("--scan_dir path does not exist or is not a directory: %s", scan_dir)
        sys.exit(1)

    # Collect candidate dataset roots (immediate sub-directories only).
    candidates = sorted(p for p in scan_dir.iterdir() if p.is_dir())
    if not candidates:
        logger.error("No sub-directories found in %s.", scan_dir)
        sys.exit(1)

    dataset_roots = [p for p in candidates if _is_dataset_root(p)]
    if not dataset_roots:
        logger.error(
            "None of the %d sub-directories in %s look like lerobot datasets "
            "(missing meta/info.json).",
            len(candidates),
            scan_dir,
        )
        sys.exit(1)

    logger.info("Found %d dataset(s) to upload in %s", len(dataset_roots), scan_dir)

    results = []
    for root in dataset_roots:
        # Infer repo_id as <parent_name>/<folder_name>
        repo_id = f"{scan_dir.name}/{root.name}"
        logger.info("--- %s (%s) ---", repo_id, root)
        ok = _upload_one(repo_id=repo_id, root=root, base_url=args.base_url, dry_run=args.dry_run)
        results.append((repo_id, ok))

    # Summary
    succeeded = [r for r, ok in results if ok]
    failed = [r for r, ok in results if not ok]
    logger.info(
        "Done. %d succeeded, %d failed.",
        len(succeeded),
        len(failed),
    )
    if failed:
        for r in failed:
            logger.error("  FAILED: %s", r)
        sys.exit(1)


if __name__ == "__main__":
    main()
