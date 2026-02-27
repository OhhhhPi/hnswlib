#!/usr/bin/env python3
"""
Download the LoCoMo dataset (locomo10.json) from GitHub.

Usage:
    python scripts/locomo/download_data.py --output_dir data/locomo
"""

import argparse
import json
import os
import urllib.request


LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)


def download_locomo(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "locomo10.json")

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        # Validate
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Contains {len(data)} samples")
        return

    print(f"Downloading LoCoMo dataset from {LOCOMO_URL} ...")
    urllib.request.urlretrieve(LOCOMO_URL, output_path)
    print(f"Saved to {output_path}")

    # Validate JSON
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n=== LoCoMo Dataset Statistics ===")
    print(f"Total samples: {len(data)}")

    total_dialogs = 0
    total_qa = 0
    categories = set()
    for sample in data:
        total_dialogs += len(sample.get("conversation", []))
        for qa in sample.get("qa_pairs", []):
            total_qa += 1
            categories.add(qa.get("category", "unknown"))

    print(f"Total dialog turns: {total_dialogs}")
    print(f"Total QA pairs: {total_qa}")
    print(f"QA categories: {sorted(categories)}")


def main():
    parser = argparse.ArgumentParser(description="Download LoCoMo dataset")
    parser.add_argument(
        "--output_dir", default="data/locomo", help="Output directory"
    )
    args = parser.parse_args()
    download_locomo(args.output_dir)


if __name__ == "__main__":
    main()
