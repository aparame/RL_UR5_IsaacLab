# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility script and module to download pretrained checkpoints from GitHub Releases.
Avoids storing heavy model weights in Git LFS or repository logs.
"""

import argparse
import os
import sys
import urllib.request
from typing import Optional

# Default repository and release tag for published checkpoints
DEFAULT_REPO = os.getenv("CHECKPOINT_REPO", "aparame/RL_UR5_Private")
DEFAULT_TAG = os.getenv("CHECKPOINT_TAG", "v1.0.0-checkpoints")

# Mapping from local relative paths (under repo root) to asset filenames in GitHub Release
CHECKPOINT_CATALOG = {
    "camera_pose_tracking_rgb_v5_60hz": {
        "path": "logs/skrl/logs/skrl_camera_pose_tracking/rgb_v5_60hz/checkpoints/best_agent.pt",
        "asset": "camera_pose_tracking_rgb_v5_60hz_best_agent.pt",
    },
    "hierarchical_depth_v1": {
        "path": "logs/skrl/logs/skrl_hierarchical_depth/v1/checkpoints/best_agent.pt",
        "asset": "hierarchical_depth_v1_best_agent.pt",
    },
    "hierarchical_depth_v2": {
        "path": "logs/skrl/logs/skrl_hierarchical_depth/v2/checkpoints/best_agent.pt",
        "asset": "hierarchical_depth_v2_best_agent.pt",
    },
    "hierarchical_rgb_v1": {
        "path": "logs/skrl/logs/skrl_hierarchical_rgb/v1/checkpoints/best_agent.pt",
        "asset": "hierarchical_rgb_v1_best_agent.pt",
    },
}


def get_repo_root() -> str:
    """Find the root directory of the repository."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, ".git")) or os.path.exists(
            os.path.join(current, ".gitattributes")
        ):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[INFO] Downloading checkpoint from: {url}")
    print(f"[INFO] Saving to: {dest_path}")

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        mb_downloaded = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        sys.stdout.write(
            f"\r Progress: {percent}% ({mb_downloaded:.1f} MB / {total_mb:.1f} MB)"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
        print("\n[INFO] Download completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Failed to download checkpoint from {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e


def ensure_checkpoint_exists(
    target_path: str,
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_TAG,
) -> str:
    """Ensure the specified checkpoint exists locally.

    If target_path is missing, try downloading from GitHub Releases.
    """
    abs_path = os.path.abspath(target_path)
    if os.path.exists(abs_path):
        return abs_path

    repo_root = get_repo_root()
    rel_path = os.path.relpath(abs_path, repo_root)

    # Search catalog for matching path
    matching_asset = None
    for item in CHECKPOINT_CATALOG.values():
        norm_catalog_path = os.path.normpath(item["path"])
        if norm_catalog_path == os.path.normpath(rel_path):
            matching_asset = item["asset"]
            break

    if not matching_asset:
        matching_asset = os.path.basename(abs_path)

    url = f"https://github.com/{repo}/releases/download/{tag}/{matching_asset}"
    try:
        download_file(url, abs_path)
    except Exception:
        # Fall back to public repository if origin fails
        if repo != "aparame/RL_UR5_IsaacLab":
            fallback_url = f"https://github.com/aparame/RL_UR5_IsaacLab/releases/download/{tag}/{matching_asset}"
            print("[INFO] Retrying download from public fallback repository...")
            download_file(fallback_url, abs_path)

    return abs_path


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained checkpoints from GitHub Releases."
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all registered catalog checkpoints."
    )
    parser.add_argument(
        "--key",
        type=str,
        choices=list(CHECKPOINT_CATALOG.keys()),
        help="Download a specific checkpoint by catalog key.",
    )
    parser.add_argument(
        "--repo", type=str, default=DEFAULT_REPO, help="GitHub repository (owner/name)."
    )
    parser.add_argument(
        "--tag", type=str, default=DEFAULT_TAG, help="GitHub release tag."
    )
    args = parser.parse_args()

    repo_root = get_repo_root()

    if args.key:
        item = CHECKPOINT_CATALOG[args.key]
        target = os.path.join(repo_root, item["path"])
        ensure_checkpoint_exists(target, repo=args.repo, tag=args.tag)
    elif args.all or len(sys.argv) == 1:
        for key, item in CHECKPOINT_CATALOG.items():
            print(f"\n--- Processing checkpoint key: {key} ---")
            target = os.path.join(repo_root, item["path"])
            try:
                ensure_checkpoint_exists(target, repo=args.repo, tag=args.tag)
            except Exception as e:
                print(f"[WARNING] Could not fetch {key}: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
