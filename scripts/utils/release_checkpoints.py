# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility script to package local model checkpoints for GitHub Releases.
"""

import argparse
import os
import shutil
import sys

from download_checkpoints import CHECKPOINT_CATALOG, get_repo_root


def main():
    parser = argparse.ArgumentParser(
        description="Package checkpoints for GitHub Release uploads."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v1.0.0-checkpoints",
        help="Release tag name (default: v1.0.0-checkpoints).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="release_assets",
        help="Directory to copy asset binaries into.",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Packaging release assets into: {output_dir}")
    collected = []

    for key, item in CHECKPOINT_CATALOG.items():
        src_path = os.path.join(repo_root, item["path"])
        if os.path.exists(src_path):
            dest_asset_path = os.path.join(output_dir, item["asset"])
            shutil.copy2(src_path, dest_asset_path)
            collected.append((key, item["asset"], src_path))
            print(f"  [+] Prepared asset: {item['asset']} (from {item['path']})")
        else:
            print(f"  [-] Local file missing for key '{key}': {src_path}")

    print("\n" + "=" * 65)
    print(" GITHUB RELEASE SETUP INSTRUCTIONS:")
    print("=" * 65)
    print(f"1. Push release tag to remotes:")
    print(f"   git tag {args.tag}")
    print(f"   git push origin {args.tag}")
    print(f"   git push public {args.tag}")
    print(f"\n2. Create a new GitHub Release on your repository web page:")
    print(f"   - Target Tag: {args.tag}")
    print(f"   - Title: Pretrained Model Checkpoints ({args.tag})")
    print(f"   - Attach the following binary files from folder '{output_dir}/':")
    for _, asset_name, _ in collected:
        print(f"     * {asset_name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
