# Hybris compaction, and how ember differs

Source: `hybris/rust/crates/pi-agent/src/compaction.rs`,
`pi-agent/src/messages.rs`, `pi-ai/src/utils/estimate.rs`, and
`pi-coding-agent/src/agent_session.rs`. This is the live path, not the
`MIDTURN_COMPACTION_PLAN.md` draft.

Ember's failures on session `20260921_235958_8f4900` are listed in
`compaction-issues.md`. This note says what hybris does at each of those
points.

## End to end

Hybris has two triggers, and both call the same `compact_now_with_reason`.

1. **After a finished run.** `maybe_auto_compact` runs at the end of `prompt`
   and from `finish_turn`. It skips a cancelled response. If the last
   assistant message has provider usage, that usage is the context size.
   Otherwise it estimates. Usage recorded before the latest compaction
   boundary is ignored, so one compaction does not immediately cause another.

2. **Mid-turn.** The agent loop's `prepare_next_turn` hook calls
   `compact_before_next_assistant_response` after an assistant response,
   before the next model request in the same tool loop. The estimate includes
   the tool results that were just appended. If the threshold is crossed,
   compaction runs and the hook returns the replaced message list. The loop
   sends that list, not the pre-compaction one.

The threshold is the model's own window, not a separate budget:

```text
context_tokens > model.context_window - reserve_tokens
```

Defaults: `reserve_tokens = 16384`, `keep_recent_tokens = 20000`,
`enabled = true`. On a 1,000,000-token model this fires only past about
983,616 tokens. `context_window == 0` skips mid-turn compaction.

`prepare_compaction` then:

- Finds the latest `compactionSummary` and summarizes only messages after it.
  The previous summary is passed through whole, inside `<previous-summary>`.
- Walks backward from the end until `keep_recent_tokens` (20,000) is covered.
- Cuts on the next message that is not a tool result. A kept region never
  starts on an orphan tool result.
- If that cut is inside a turn, the turn splits. History before the turn is
  one summary. The dropped prefix of the current turn is a second summary,
  appended as `Turn Context (split turn)`. The suffix of the turn is kept
  verbatim.

`serialize_conversation` gives the summarizer the actual turns:

- user text
- assistant thinking, assistant text, and tool calls as `name(args)`
- tool results, each capped at 2,000 UTF-16 units, with
  `[... N more characters truncated]` on the end

The summary call has its own system prompt: do not continue the conversation,
only emit the checkpoint. The first pass must use Goal, Constraints, Progress
(Done / In Progress / Blocked), Key Decisions, Next Steps, Critical Context,
and must keep exact paths, function names, and error messages. Later passes
use the update prompt, whose first rule is to preserve the previous summary.
Output is capped at `floor(0.8 * reserve_tokens)` (13,107 tokens by default),
and again at the model's `max_tokens`. A split-turn prefix gets a smaller
budget, half of `reserve_tokens`. An empty or failed summary returns an error
and does not replace the transcript.

File paths from `read`, `write`, and `edit` calls are collected and appended
as `<read-files>` and `<modified-files>`. Those lists are also stored on the
summary so the next pass still has them.

The live context becomes one `compactionSummary` custom message plus the kept
tail. The session file is append-only: a `type: compaction` entry records the
summary, `tokensBefore`, `firstKeptEntryId`, and the retained tail. The old
messages stay in the file.

At request time only, `convert_to_llm` renders that custom message as a user
message wrapped like this:

```text
The conversation history before this point was compacted into the following summary:

<summary>
...
</summary>
```

There is no second writer. Compaction does not append a memory file.

## Against ember

| | Hybris | Ember |
| --- | --- | --- |
| When | After a run, and again inside a tool loop before the next model call | Once, at the start of a user turn. Mid-loop only if the provider returns a context-overflow error |
| Window | `model.context_window` | Fixed 32,000, ignoring the provider |
| Threshold | usage > window − 16,384 | estimate > 80% of 32,000 |
| Size signal | Last assistant usage, plus chars/4 only for messages after it | chars/4 for every message. System prompt and tool schemas omitted. Echoed reasoning included |
| What is kept | ~20,000 tokens, aligned off tool results | Last 6 messages, wherever that falls |
| Tool pairs | Kept region cannot start on a tool result. A split turn summarizes the dropped prefix instead of leaving the result behind | Cut can fall between a call and its result. Repair drops the result from memory and leaves it in the session file |
| Previous summary | Passed in full, with an instruction to preserve it | Re-clipped to 1,000 characters and summarized again as ordinary text |
| Summarizer input | Thinking, text, `name(args)`, tool results with a truncation marker | 1,000 characters of `content` only. No marker, no tool name, no arguments, no reasoning. Blank assistant turns stay blank |
| Summary shape | Fixed sections. Told not to answer the conversation | "Summarize concisely." No faithfulness check |
| Summary length | Capped near 13k tokens | Uncapped. The three ember calls wrote 2,806, 2,901, and 1,367 completion tokens, with thinking left on and discarded |
| Stored as | `compactionSummary`, rendered as a labeled summary only when sending | `role: user`, so the summary's own orders ("do NOT re-read") look like the user |
| Transcript | Append a compaction entry. Old messages remain | Rewrite the session JSONL. The pre-cut messages exist only in the trace |
| Failure | Leave the messages unchanged | Replace the session with message 0 plus the last 6 |
| Side record | File paths on the summary | A second LLM call appends `memory/<date>.md` from a different slice. It disagreed with the summary on birth time, and that file is not in the prompt |

## What this would have changed on the ember session

The three cuts fired at summary prompts of 7,965, 4,795, and 937 tokens.
Under hybris those would not have fired on a 1M model. The Grok answer would
not have been marked "never delivered" while sitting in the kept tail, because
the kept tail is part of the token budget the cutter can see, and a previous
summary is updated rather than clipped. The poem line would not have become
*"her e…" (truncated)*, because that string was a 1,000-character clip of an
older summary with no truncation marker. The orphan `web_fetch` at the top of
the session file would not have been written: the cut is not allowed to start
on a tool result, and a failed or partial compaction does not rewrite the file.

Hybris still estimates trailing messages at about 4 characters per token, and
it still shows the model a summary in the user role. The wrapper text and the
stored `compactionSummary` role are what stop that summary from being just
another user turn.
