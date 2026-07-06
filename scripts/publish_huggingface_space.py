"""Create and publish the PnC visualizer as a Hugging Face Docker Space."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_NAME = "pnc-policy-visualizer"
UPLOAD_PATTERNS = [
    "README.md",
    "Dockerfile",
    "requirements-web.txt",
    "src/**",
    "data/pnc_model/pnc_mouse_tracked_26182EVZW/env_info.yaml",
    "data/pnc_model/pnc_mouse_tracked_26182EVZW/best/best_model.zip",
    "data/pnc_model/pnc_mouse_tracked_26182EVZW/best/visualizer_best_params.yaml",
    "data/pnc_model/pnc_mouse_tracked_26182EVZW/visualizer_last_params.yaml",
    "data/pnc_model/pnc_mouse_tracked_26182EVZW/last/last_model.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the PnC visualizer online.")
    parser.add_argument("--space-name", default=DEFAULT_SPACE_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("A browser window will open for secure Hugging Face authentication.")
    login(skip_if_logged_in=True)
    api = HfApi()
    user = api.whoami()["name"]

    space_name = args.space_name.strip() or DEFAULT_SPACE_NAME
    repo_id = space_name if "/" in space_name else f"{user}/{space_name}"

    print(f"Creating Docker Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("Uploading the model and visualizer files...")
    api.upload_folder(
        folder_path=PROJECT_ROOT,
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=UPLOAD_PATTERNS,
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="Deploy standalone PnC policy visualizer",
    )
    print(f"\nPublished: https://huggingface.co/spaces/{repo_id}")
    print("The first Docker build can take several minutes.")


if __name__ == "__main__":
    main()
