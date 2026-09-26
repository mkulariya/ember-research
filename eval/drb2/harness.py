"""DeepResearch Bench II harness for ember.

    python eval/drb2/fetch.py                                  # once
    python eval/drb2/harness.py run   --subset local7 --name baseline
    python eval/drb2/harness.py grade --name baseline         # judge: JUDGE_API_KEY
    python eval/drb2/harness.py score --name baseline
    python eval/drb2/harness.py compare baseline next

`run` executes ember once per task in its own process and workspace, writes
reports/idx-N.md plus the trace. `grade` scores each report with DRB2's own
prompt and parser, using an OpenAI-compatible judge (default deepseek-flash).
Everything lands in eval/runs/<name>/; finished tasks are skipped on rerun.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
UPSTREAM = HERE / "upstream"
RUNS = Path(os.environ.get("EMBER_EVAL_RUNS", REPO / "eval" / "runs"))

# Interactive or unsafe under yolo; a benchmark run has no one to answer them.
REMOVED_TOOLS = ("exec", "clarify")


def load_tasks() -> dict[int, dict]:
    path = UPSTREAM / "tasks_and_rubrics.jsonl"
    if not path.exists():
        sys.exit("missing DRB2 data: run `python eval/drb2/fetch.py` first")
    rows = (json.loads(line) for line in path.open(encoding="utf-8"))
    return {r["idx"]: r for r in rows}


def load_subset(name: str) -> list[int]:
    return [t["idx"] for t in json.loads((HERE / "subsets" / f"{name}.json").read_text())]


def git_state() -> dict:
    def git(*cmd: str) -> str:
        out = subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True)
        return out.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


# --- run ------------------------------------------------------------------


def _norm_url(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url.strip().lower()).rstrip("/")


def _guard_web_fetch(core, web_tools, blocked_urls: list[str]) -> None:
    """Refuse the task's own source document, which DRB2 scores as -1."""
    blocked = [_norm_url(u) for u in blocked_urls if u]
    tool = core.TOOLS["web_fetch"]
    original = tool.handler

    def guarded(args, ctx):
        urls = web_tools._as_url_list(args)
        hits = [u for u in urls if any(_norm_url(u).startswith(b) for b in blocked)]
        if not hits:
            return original(args, ctx)
        note = f"[blocked for this task, use other sources: {', '.join(hits)}]"
        allowed = [u for u in urls if u not in hits]
        if not allowed:
            return core.ToolResult(f"web_fetch: {note}", is_error=True)
        result = original({**args, "urls": allowed, "url": None}, ctx)
        return core.ToolResult(f"{note}\n{result.content}", is_error=result.is_error)

    tool.handler = guarded


def _trace_metrics(trace_path: Path) -> dict:
    m = {"outcome": "no_trace", "steps": 0, "llm_calls": 0, "aux_llm_calls": 0,
         "prompt_tokens": 0, "completion_tokens": 0, "compactions": 0,
         "tool_errors": 0, "tools": {}}
    if not trace_path.exists():
        return m
    tools: collections.Counter = collections.Counter()
    for line in trace_path.open(encoding="utf-8"):
        e = json.loads(line)
        kind = e.get("type")
        if kind == "llm_call":
            usage = e.get("usage") or {}
            if e.get("purpose") == "agent":
                m["llm_calls"] += 1
                m["prompt_tokens"] += usage.get("prompt_tokens") or 0
                m["completion_tokens"] += usage.get("completion_tokens") or 0
            else:
                m["aux_llm_calls"] += 1
        elif kind == "tool":
            tools[e.get("name")] += 1
            m["tool_errors"] += bool(e.get("is_error"))
        elif kind == "context_edit" and e.get("kind") == "compaction":
            m["compactions"] += 1
        elif kind == "turn_end":
            m["outcome"] = e.get("outcome")
            m["steps"] = e.get("step", 0)
    m["tools"] = dict(tools)
    return m


def run_task(a: argparse.Namespace) -> dict:
    """One task, in-process. Called in a child process by `run`."""
    sys.path.insert(0, str(REPO))
    import ember.core as core
    from ember import web_tools

    task = load_tasks()[a.idx]
    run_dir = RUNS / a.name
    for sub in ("reports", "metrics", "traces"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    for name in REMOVED_TOOLS:
        core.TOOLS.pop(name, None)
    _guard_web_fetch(core, web_tools, task["content"].get("blocked", {}).get("urls", []))

    config = core.Config.from_env()
    config.api_base_url = a.agent_base_url
    config.api_key = a.agent_api_key
    config.model = a.agent_model
    config.context_window = a.context_window
    config.max_steps = a.max_steps
    config.workspace = str(run_dir / "work" / f"idx-{a.idx}")
    config.yolo = True

    agent, _ = core.bootstrap(config)
    agent.start_session()
    t0 = time.time()
    # The DRB2 task text is sent as-is: an added "final answer" instruction
    # made the model skip tools and write from memory.
    reply = agent.run_turn(task["content"]["task"])
    elapsed = time.time() - t0

    trace = Path(config.workspace) / ".traces" / f"{agent.session_id}.jsonl"
    metrics = {"idx": a.idx, "wall_s": round(elapsed, 1), **_trace_metrics(trace)}
    if trace.exists():
        (run_dir / "traces" / f"idx-{a.idx}.jsonl").write_bytes(trace.read_bytes())
    if metrics["outcome"] == "final" and reply.strip():
        (run_dir / "reports" / f"idx-{a.idx}.md").write_text(reply, encoding="utf-8")
    (run_dir / "metrics" / f"idx-{a.idx}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def cmd_run(a: argparse.Namespace) -> None:
    run_dir = RUNS / a.name
    for sub in ("reports", "metrics", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    idxs = a.only or load_subset(a.subset)
    (run_dir / "config.json").write_text(json.dumps({
        "subset": a.subset, "idxs": idxs, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(), "agent": {k: v for k, v in vars(a).items()
                                      if k.startswith(("agent_", "context", "max_", "task_"))
                                      and k != "agent_api_key"},
    }, indent=2))

    todo = [i for i in idxs if not (run_dir / "metrics" / f"idx-{i}.json").exists()]
    print(f"{a.name}: {len(idxs)} tasks, {len(idxs) - len(todo)} done, running {len(todo)} "
          f"with {a.workers} worker(s)")

    def launch(idx: int) -> None:
        cmd = [sys.executable, __file__, "run-one", "--name", a.name, "--idx", str(idx),
               "--agent-model", a.agent_model, "--agent-base-url", a.agent_base_url,
               "--agent-api-key", a.agent_api_key, "--context-window", str(a.context_window),
               "--max-steps", str(a.max_steps)]
        t0 = time.time()
        with (run_dir / "logs" / f"idx-{idx}.log").open("w") as log:
            try:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.task_timeout)
            except subprocess.TimeoutExpired:
                (run_dir / "metrics" / f"idx-{idx}.json").write_text(json.dumps(
                    {"idx": idx, "outcome": "timeout", "wall_s": a.task_timeout}))
        mfile = run_dir / "metrics" / f"idx-{idx}.json"
        outcome = json.loads(mfile.read_text())["outcome"] if mfile.exists() else "crashed"
        if not mfile.exists():
            mfile.write_text(json.dumps({"idx": idx, "outcome": "crashed"}))
        print(f"  idx-{idx}: {outcome} in {time.time() - t0:.0f}s")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(launch, todo))


# --- grade ----------------------------------------------------------------


def _upstream():
    sys.path.insert(0, str(UPSTREAM))
    import aggregate_scores
    import gpt_client
    import run_evaluation
    return run_evaluation, gpt_client, aggregate_scores


class Judge:
    """Stands in for DRB2's GPT-5.5 client; same query() contract."""

    def __init__(self, base_url: str, model: str, api_key: str, max_tokens: int, output_cls):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=900, max_retries=2)
        self.model = model
        self.max_tokens = max_tokens
        self.output_cls = output_cls

    def complete(self, prompt: str) -> tuple[str, dict]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        return resp.choices[0].message.content or "", {
            "promptTokenCount": usage.prompt_tokens,
            "candidatesTokenCount": usage.completion_tokens,
            "totalTokenCount": usage.total_tokens,
            "thoughtsTokenCount": getattr(details, "reasoning_tokens", 0) or 0,
        }

    def query(self, input_data):
        text, usage = self.complete(input_data.text)
        return self.output_cls(text=_pretty_json(text), usage_metadata=usage)


def _pretty_json(text: str) -> str:
    """DRB2's parser escapes quotes with a per-line regex that breaks on
    single-line JSON, so hand it the judge's JSON re-dumped with one key per
    line. Text that is not JSON passes through for DRB2 to reject and retry."""
    fenced = re.search(r"```json\s*(.*)```", text, re.DOTALL)
    try:
        obj = json.loads(fenced.group(1) if fenced else text)
    except json.JSONDecodeError:
        return text
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def cmd_grade(a: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO))
    import ember.core  # noqa: F401  -- loads .env into the environment

    api_key = os.environ.get("JUDGE_API_KEY")
    if not api_key:
        sys.exit("set JUDGE_API_KEY (in .env or the environment) for the judge")
    run_evaluation, gpt_client, _ = _upstream()
    run_evaluation.client = Judge(a.judge_base_url, a.judge_model, api_key,
                                  a.judge_max_tokens, gpt_client.GPTOutput)

    run_dir = RUNS / a.name
    grades = run_dir / "grades"
    grades.mkdir(exist_ok=True)
    tasks = load_tasks()
    reports = sorted((run_dir / "reports").glob("idx-*.md"))
    todo = [p for p in reports if a.regrade or not (grades / f"{p.stem}.json").exists()]
    print(f"{a.name}: grading {len(todo)} of {len(reports)} reports with {a.judge_model}")

    def grade(path: Path) -> None:
        idx = int(path.stem.split("-")[1])
        _, result, _ = run_evaluation.process_one_with_chunking(
            idx, str(path), tasks[idx]["content"], a.chunk_size, 150000, a.max_retries)
        result["judge"] = {"model": a.judge_model, "base_url": a.judge_base_url}
        (grades / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(grade, todo))
    cmd_score(a)


# --- score / compare --------------------------------------------------------


DIMS = ("total", "inforecall", "analysis", "presentation", "blocked_rate")


def build_scorecard(name: str, in_price: float, out_price: float) -> dict:
    _, _, aggregate_scores = _upstream()
    run_dir = RUNS / name
    config = json.loads((run_dir / "config.json").read_text())
    tasks = load_tasks()
    rows, judge_in, judge_out = [], 0, 0
    for idx in config["idxs"]:
        mfile = run_dir / "metrics" / f"idx-{idx}.json"
        gfile = run_dir / "grades" / f"idx-{idx}.json"
        row = {"idx": idx, "theme": tasks[idx]["theme"],
               **(json.loads(mfile.read_text()) if mfile.exists() else {"outcome": "not_run"})}
        grade = json.loads(gfile.read_text()) if gfile.exists() else None
        if grade and "error" not in grade:
            row.update(aggregate_scores.compute_dimension_averages(grade))
            judge_in += grade["usage_summary"]["input_tokens"]
            judge_out += grade["usage_summary"]["output_tokens"]
        else:
            # No report or failed grading counts as zero; the reason stays visible.
            row.update({d: 0.0 for d in DIMS})
            row["graded"] = False if grade is None else "error"
        rows.append(row)
    n = len(rows) or 1
    return {
        "name": name, "config": config, "tasks": rows,
        "mean": {d: round(sum(r.get(d) or 0 for r in rows) / n, 4) for d in DIMS},
        "judge_tokens": {"input": judge_in, "output": judge_out},
        "judge_cost_usd_est": round(judge_in / 1e6 * in_price + judge_out / 1e6 * out_price, 4),
    }


def cmd_score(a: argparse.Namespace) -> None:
    card = build_scorecard(a.name, a.judge_in_price, a.judge_out_price)
    (RUNS / a.name / "scorecard.json").write_text(json.dumps(card, indent=2))
    print(f"\n{a.name}  ({card['config']['subset']}, git {card['config']['git']['commit'][:8]}"
          f"{' dirty' if card['config']['git']['dirty'] else ''})")
    print(f"{'idx':>4} {'outcome':<14} {'total':>6} {'recall':>6} {'anal':>6} {'pres':>6} "
          f"{'blk':>5} {'steps':>5} {'wall_s':>7}  theme")
    for r in card["tasks"]:
        print(f"{r['idx']:>4} {r.get('outcome', ''):<14} {r['total']:>6.3f} {r['inforecall'] or 0:>6.3f} "
              f"{r['analysis'] or 0:>6.3f} {r['presentation'] or 0:>6.3f} {r['blocked_rate'] or 0:>5.2f} "
              f"{r.get('steps', 0):>5} {r.get('wall_s', 0):>7}  {r['theme']}")
    m = card["mean"]
    print(f"mean {'':<14} {m['total']:>6.3f} {m['inforecall']:>6.3f} {m['analysis']:>6.3f} "
          f"{m['presentation']:>6.3f} {m['blocked_rate']:>5.2f}")
    print(f"judge tokens in/out {card['judge_tokens']['input']}/{card['judge_tokens']['output']}"
          f"  est ${card['judge_cost_usd_est']}")


def cmd_compare(a: argparse.Namespace) -> None:
    base = json.loads((RUNS / a.base / "scorecard.json").read_text())
    new = json.loads((RUNS / a.new / "scorecard.json").read_text())
    new_rows = {r["idx"]: r for r in new["tasks"]}
    print(f"{'idx':>4} {a.base[:10]:>10} {a.new[:10]:>10} {'delta':>7}  theme")
    for r in base["tasks"]:
        other = new_rows.get(r["idx"])
        if other:
            print(f"{r['idx']:>4} {r['total']:>10.3f} {other['total']:>10.3f} "
                  f"{other['total'] - r['total']:>+7.3f}  {r['theme']}")
    for d in DIMS:
        print(f"{d:<13} {base['mean'][d]:.3f} -> {new['mean'][d]:.3f}  ({new['mean'][d] - base['mean'][d]:+.3f})")


# --- cli --------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="DeepResearch Bench II harness for ember")
    sub = p.add_subparsers(dest="cmd", required=True)

    def agent_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--name", required=True)
        sp.add_argument("--agent-model", default="lfm25-thinking-tools:Q8_0")
        sp.add_argument("--agent-base-url", default="http://localhost:11434/v1")
        sp.add_argument("--agent-api-key", default="ollama")
        sp.add_argument("--context-window", type=int, default=32000)
        sp.add_argument("--max-steps", type=int, default=100)

    run = sub.add_parser("run", help="run ember on a subset")
    agent_args(run)
    run.add_argument("--subset", default="local7", choices=["local7", "kaggle25"])
    run.add_argument("--only", type=int, nargs="*", help="task idxs instead of the subset")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--task-timeout", type=int, default=3600)

    one = sub.add_parser("run-one", help="internal: one task in this process")
    agent_args(one)
    one.add_argument("--idx", type=int, required=True)

    def price_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--name", required=True)
        # DeepSeek V4.1 Flash off-peak, $/1M tokens (cache-miss input).
        sp.add_argument("--judge-in-price", type=float, default=0.15)
        sp.add_argument("--judge-out-price", type=float, default=0.60)

    grade = sub.add_parser("grade", help="grade reports with the judge")
    price_args(grade)
    grade.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "deepseek-flash"))
    grade.add_argument("--judge-base-url",
                       default=os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com"))
    grade.add_argument("--judge-max-tokens", type=int, default=32768)
    grade.add_argument("--chunk-size", type=int, default=50)
    grade.add_argument("--max-retries", type=int, default=3)
    grade.add_argument("--workers", type=int, default=8)
    grade.add_argument("--regrade", action="store_true")

    score = sub.add_parser("score", help="print the scorecard")
    price_args(score)

    cmp_ = sub.add_parser("compare", help="compare two scored runs")
    cmp_.add_argument("base")
    cmp_.add_argument("new")

    a = p.parse_args()
    {"run": cmd_run, "run-one": run_task, "grade": cmd_grade,
     "score": cmd_score, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
