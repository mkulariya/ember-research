# Compaction issues

Session inspected: `20260921_235958_8f4900`, model `deepseek-flash`.
Compaction ran three times. Code is `compact_history`, `_format_summary_block`,
`flush_conversation`, and the soft-limit check in `Agent.run_turn`
(`ember/core.py`).

Each pass lost information. The third pass started treating that loss as fact.

## 1. The cut fires for the wrong reason

The trigger is ember's own estimate: 80% of a 32,000-token window, about one
character per four tokens (`EMBER_CONTEXT_WINDOW`, default 32,000,
`compact_threshold` 0.8). DeepSeek-V4.1-Flash has a 1M window. The three
summary calls were 7,965, 4,795, and 937 prompt tokens. The agent call right
after the second cut was 11,819 tokens. History was compressed because the
local counter filled up, not because the provider was full.

The estimate ignores the system prompt and the tool schemas, and it counts
echoed reasoning. The number that dominates is untruncated tool bodies.

## 2. Fat tool results are kept, and the summary is what gets thrown away

A tool result stays whole until it passes 38,400 characters
(`context_window * 0.3 * 4`). Compaction runs first, at a lower budget, so a
few fetches of 10k–24k characters force a cut while truncation never runs.

`keep_fresh` is 6. The last six messages are the ones kept, and those are the
newest fetches. The summary is message 0, so it is the first thing the next
cut destroys.

## 3. The slice is not a turn boundary

The keep count is in messages, with no walk back to a user turn or a finished
tool call. All three cuts landed between a tool call and its results.

The repair then dropped orphan tool results from the live context: 1, then 2,
then 1. The session file is rewritten before that repair, so the orphans stay
on disk. After the third cut the file still started with a `web_fetch` result,
`call_00_OJuXQyBB`, that has no call in front of it. A resumed session loads
that invalid history. Repair mutates memory only. It does not rewrite the file.

## 4. The summarizer never sees the conversation

`_format_summary_block` keeps 1,000 characters of each message and does not
mark that the rest was cut. Tool-call names and arguments are omitted.
Reasoning is omitted. Empty tool messages are skipped. Assistant turns with
no content are included as a blank `[assistant]`.

In the first pass, 54 messages became a 26,737-character block: 15 of 21
assistant turns were blank, and 24 of 54 chunks were sliced mid-sentence.
The model summarized heads of tool dumps and empty turns.

## 5. The summary contradicts the messages kept beside it

The summarizer does not see the last six messages.

- The first summary says the Grok 4.7 answer was never delivered. That answer
  had already been sent (15,347 characters) and was sitting in the kept tail.
- The second summary says no birth data is known. The next message is
  `20.02.1995, Degana, rajasthan.`

## 6. Each new summary is built from a damaged copy of the previous one

The previous summary is fed back through the same 1,000-character clip. By
the third pass:

- The poem's closing line, which both earlier summaries had in full, is
  recorded as starting *"her e…" (truncated)*.
- The eight-part poem is reduced to "exists (contents not restated)."
- The retraction's subject is "unspecified in compacted text."

The clip became a fact about the poem.

## 7. The summary is stored as a user message

`_compaction_message` uses `role="user"` and is placed first. Lines the
summarizer wrote ("do NOT re-read", "do NOT fix the poem", "next turn must
address them") arrive as user instructions.

It also copies claims out of files it read. `AGENTS.md` and `MEMORY.md` say
they are injected every turn. They are not. `build_system_prompt` does not
read them. That claim is now session context.

## 8. The same `compact()` call writes a second, disagreeing record

Before the summary, `flush_conversation` asks the same model for durable
bullets and appends them to `memory/YYYY-MM-DD.md`.

That prompt is a different slice: user and assistant text only, clipped at
2,000 characters, including the tail the summarizer does not see, and
excluding tool results. It runs on every compaction and appends another
`## 2026-09-22` section, so the same decisions are duplicated.

The third flush recorded a birth time of about 1:00 AM. The summary of that
same cut says no birth time was given. The user message was the date and the
place.

`memory/<date>.md` is not put into the system prompt. The next turn does not
see it unless the model searches memory.

## 9. The summary model thinks, and the thought is thrown away

Summary and memory-flush calls use `complete_plain` on the normal chat model.
Thinking stays on. Only `content` is kept.

The third memory flush spent 1,690 prompt tokens and came back with 6,018
completion tokens and about 21,000 characters of reasoning, to produce a
bullet list.

## 10. A failed summary deletes the middle

If the summary call throws, the replacement is `messages[0]` plus the last
six messages. Everything between is dropped, and the session file is rewritten
that way. Before any successful compaction, `messages[0]` is the first user
turn ("meow"), not a summary.

There is no check that the summary is faithful, complete, or consistent with
the kept tail.
