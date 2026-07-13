# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Upload release assets directly to GitHub Releases via GitHub REST API.
"""

import json
import os
import urllib.parse
import urllib.request

def get_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        return token
    cred_file = os.path.expanduser("~/.git-credentials")
    if os.path.exists(cred_file):
        with open(cred_file, "r") as f:
            for line in f:
                if "github.com" in line and ":" in line and "@" in line:
                    token_part = line.split("@")[0].split(":")[-1]
                    if token_part:
                        return token_part
    return ""

TOKEN = get_github_token()
TAG = "v1.0.0-checkpoints"
REPOS = ["aparame/RL_UR5_Private", "aparame/RL_UR5_IsaacLab"]

ASSETS = [
    ("camera_pose_tracking_rgb_v5_60hz_best_agent.pt", "release_assets/camera_pose_tracking_rgb_v5_60hz_best_agent.pt"),
    ("hierarchical_depth_v1_best_agent.pt", "release_assets/hierarchical_depth_v1_best_agent.pt"),
    ("hierarchical_depth_v2_best_agent.pt", "release_assets/hierarchical_depth_v2_best_agent.pt"),
    ("hierarchical_rgb_v1_best_agent.pt", "release_assets/hierarchical_rgb_v1_best_agent.pt"),
]


def make_request(url, method="GET", data=None, headers=None):
    if headers is None:
        headers = {}
    headers["Authorization"] = f"token {TOKEN}"
    headers["User-Agent"] = "RL_UR5-Release-Uploader"
    headers["Accept"] = "application/vnd.github.v3+json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_asset(upload_url_template, filename, filepath):
    base_upload_url = upload_url_template.split("{")[0]
    query = urllib.parse.urlencode({"name": filename})
    target_url = f"{base_upload_url}?{query}"

    print(f"  --> Uploading {filename} ({os.path.getsize(filepath) / (1024*1024):.1f} MB)...")
    headers = {
        "Authorization": f"token {TOKEN}",
        "User-Agent": "RL_UR5-Release-Uploader",
        "Content-Type": "application/octet-stream",
    }

    with open(filepath, "rb") as f:
        file_data = f.read()

    req = urllib.request.Request(target_url, data=file_data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"      Upload complete! Asset ID: {res.get('id')}")


def main():
    for repo in REPOS:
        print(f"\n=======================================================")
        print(f" Processing GitHub Release for repository: {repo}")
        print(f"=======================================================")
        
        release = None
        try:
            get_url = f"https://api.github.com/repos/{repo}/releases/tags/{TAG}"
            release = make_request(get_url)
            print(f"[INFO] Found existing release for tag '{TAG}' (ID: {release['id']})")
        except Exception:
            print(f"[INFO] Creating new release '{TAG}'...")
            create_url = f"https://api.github.com/repos/{repo}/releases"
            payload = json.dumps({
                "tag_name": TAG,
                "name": f"Pretrained Model Checkpoints ({TAG})",
                "body": "Pretrained PyTorch model checkpoints for RL_UR5 tasks.",
                "draft": False,
                "prerelease": False
            }).encode("utf-8")
            try:
                release = make_request(create_url, method="POST", data=payload, headers={"Content-Type": "application/json"})
                print(f"[INFO] Successfully created release (ID: {release['id']})")
            except Exception as e:
                print(f"[INFO] Creation failed ({e}), attempting to fetch release by tag...")
                try:
                    get_url = f"https://api.github.com/repos/{repo}/releases/tags/{TAG}"
                    release = make_request(get_url)
                    print(f"[INFO] Retrieved existing release (ID: {release['id']})")
                except Exception as ex:
                    print(f"[ERROR] Failed to fetch release on {repo}: {ex}")
                    continue

        upload_url = release["upload_url"]
        existing_assets = {a["name"]: a["id"] for a in release.get("assets", [])}

        for asset_name, relative_path in ASSETS:
            abs_path = os.path.abspath(relative_path)
            if not os.path.exists(abs_path):
                print(f"[WARNING] Local asset missing: {abs_path}")
                continue

            if asset_name in existing_assets:
                print(f"  [=] Asset '{asset_name}' already uploaded to release.")
            else:
                try:
                    upload_asset(upload_url, asset_name, abs_path)
                except Exception as e:
                    print(f"  [!] Failed to upload {asset_name}: {e}")

if __name__ == "__main__":
    main()
