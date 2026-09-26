"""Download DeepResearch Bench II files at a pinned commit.

Data is CC BY 4.0 (a few tasks CC BY-NC 4.0 / CC0); code is Apache 2.0.
Source: https://github.com/imlrz/DeepResearch-Bench-II
"""

import urllib.request
from pathlib import Path

COMMIT = "b38f360603db9531b102aef8c166cedb8509b6f6"
BASE = f"https://raw.githubusercontent.com/imlrz/DeepResearch-Bench-II/{COMMIT}"
FILES = ["tasks_and_rubrics.jsonl", "run_evaluation.py", "gpt_client.py", "aggregate_scores.py"]
DEST = Path(__file__).parent / "upstream"


def main() -> None:
    DEST.mkdir(exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists():
            print(f"have  {name}")
            continue
        urllib.request.urlretrieve(f"{BASE}/{name}", target)
        print(f"fetched {name}")


if __name__ == "__main__":
    main()
