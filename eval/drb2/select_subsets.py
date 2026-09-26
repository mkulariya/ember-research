"""Pick the fixed DRB2 eval subsets. Run once; the output files are committed.

kaggle25: 25 English tasks, stratified by theme (largest-remainder allocation
          proportional to theme size), seeded random pick within each theme.
local7:   7 tasks inside kaggle25, one from each of the 7 largest themes,
          the one whose rubric count is closest to the English median.
"""

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

SEED = 42
HERE = Path(__file__).parent
TASKS = HERE / "upstream" / "tasks_and_rubrics.jsonl"
OUT = HERE / "subsets"


def rubric_count(task: dict) -> int:
    return sum(len(v) for v in task["content"]["rubric"].values())


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    n = sum(sizes.values())
    quotas = {t: total * s / n for t, s in sizes.items()}
    alloc = {t: int(q) for t, q in quotas.items()}
    leftover = total - sum(alloc.values())
    # Largest remainder; ties broken by theme size, then name, for determinism.
    order = sorted(quotas, key=lambda t: (-(quotas[t] - alloc[t]), -sizes[t], t))
    for t in order[:leftover]:
        alloc[t] += 1
    return alloc


def summary(task: dict) -> dict:
    return {
        "idx": task["idx"],
        "id": task["id"],
        "theme": task["theme"],
        "description": task["description"],
        "rubrics": rubric_count(task),
    }


def main() -> None:
    rows = [json.loads(line) for line in TASKS.open(encoding="utf-8")]
    english = [r for r in rows if r["language"] == "en"]

    by_theme: dict[str, list[dict]] = defaultdict(list)
    for r in english:
        by_theme[r["theme"]].append(r)
    for tasks in by_theme.values():
        tasks.sort(key=lambda r: r["idx"])

    rng = random.Random(SEED)
    alloc = allocate({t: len(v) for t, v in by_theme.items()}, 25)
    kaggle = []
    for theme in sorted(by_theme):
        kaggle += rng.sample(by_theme[theme], alloc[theme])
    kaggle.sort(key=lambda r: r["idx"])

    median = statistics.median(rubric_count(r) for r in english)
    top_themes = sorted(by_theme, key=lambda t: (-len(by_theme[t]), t))[:7]
    local = []
    for theme in top_themes:
        picked = [r for r in kaggle if r["theme"] == theme]
        local.append(min(picked, key=lambda r: (abs(rubric_count(r) - median), r["idx"])))
    local.sort(key=lambda r: r["idx"])

    OUT.mkdir(exist_ok=True)
    for name, subset in (("kaggle25", kaggle), ("local7", local)):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps([summary(r) for r in subset], indent=2) + "\n")
        print(f"{name}: {len(subset)} tasks, {sum(rubric_count(r) for r in subset)} rubrics -> {path}")


if __name__ == "__main__":
    main()
