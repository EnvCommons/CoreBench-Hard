"""Download and prepare CORE-Bench Hard data for the OpenReward environment.

Downloads task metadata from siegelz/core-bench on HuggingFace. The test split
is GPG-encrypted and decrypted with passphrase "reproducibility".

Capsule tarballs are downloaded from corebench.cs.princeton.edu and staged in
sandbox_data/{capsule_id}/ for bucket mounting.

Usage:
    uv run python prepare_data.py [--skip-tarballs]
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

OUTPUT_DIR = Path(__file__).parent
CAPSULE_URL = "https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"


def download_tarball(capsule_id: str, dest_dir: Path, max_retries: int = 3) -> bool:
    """Download a capsule tarball to dest_dir/{capsule_id}.tar.gz."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{capsule_id}.tar.gz"

    if dest_file.exists():
        return True

    url = CAPSULE_URL.format(capsule_id=capsule_id)
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest_file)
            return True
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if dest_file.exists():
                dest_file.unlink()

    return False


def main():
    skip_tarballs = "--skip-tarballs" in sys.argv

    print("=== CORE-Bench Hard Data Preparation ===\n")

    # Download train (unencrypted)
    print("Downloading train metadata...")
    train_path = hf_hub_download("siegelz/core-bench", "core_train.json", repo_type="dataset")
    with open(train_path) as f:
        train_data = json.load(f)
    print(f"  {len(train_data)} capsules")

    # Download and decrypt test
    print("Downloading test metadata (encrypted)...")
    enc_path = hf_hub_download("siegelz/core-bench", "core_test.json.gpg", repo_type="dataset")
    dec_path = str(OUTPUT_DIR / "core_test_decrypted.json")
    subprocess.run(
        ["gpg", "--batch", "--yes", "--passphrase", "reproducibility",
         "--output", dec_path, "--decrypt", enc_path],
        check=True, capture_output=True,
    )
    with open(dec_path) as f:
        test_data = json.load(f)
    print(f"  {len(test_data)} capsules (decrypted)")

    # Build task records
    def build_tasks(capsules: list[dict]) -> list[dict]:
        tasks = []
        for capsule in capsules:
            tasks.append({
                "id": capsule["capsule_id"],
                "field": capsule["field"],
                "language": capsule["language"],
                "task_prompt": capsule["task_prompt"],
                "results": capsule["results"],
            })
        tasks.sort(key=lambda t: t["id"])
        return tasks

    train_tasks = build_tasks(train_data)
    test_tasks = build_tasks(test_data)

    # Save task JSONs
    with open(OUTPUT_DIR / "tasks_train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    with open(OUTPUT_DIR / "tasks_test.json", "w") as f:
        json.dump(test_tasks, f, indent=2)

    # Clean up decrypted file
    Path(dec_path).unlink(missing_ok=True)

    # Download capsule tarballs
    if not skip_tarballs:
        all_tasks = train_tasks + test_tasks
        print(f"\nDownloading {len(all_tasks)} capsule tarballs...")
        sandbox_dir = OUTPUT_DIR / "sandbox_data"
        success = 0
        failed = []
        for i, task in enumerate(all_tasks):
            capsule_id = task["id"]
            dest = sandbox_dir / capsule_id
            print(f"  [{i+1}/{len(all_tasks)}] {capsule_id}...", end=" ", flush=True)
            if download_tarball(capsule_id, dest):
                size_mb = (dest / f"{capsule_id}.tar.gz").stat().st_size / 1e6
                print(f"{size_mb:.1f}MB")
                success += 1
            else:
                print("FAILED")
                failed.append(capsule_id)

        print(f"\n  Downloaded: {success}/{len(all_tasks)}")
        if failed:
            print(f"  Failed: {failed}")
    else:
        print("\nSkipping tarball downloads (--skip-tarballs)")

    # Summary
    from collections import Counter
    print(f"\nDone!")
    print(f"  tasks_train.json: {len(train_tasks)} capsules")
    print(f"  tasks_test.json: {len(test_tasks)} capsules")
    print(f"  Train fields: {Counter(t['field'] for t in train_tasks)}")
    print(f"  Train languages: {Counter(t['language'] for t in train_tasks)}")


if __name__ == "__main__":
    main()
