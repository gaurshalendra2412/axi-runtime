# The ruled-out causes test: does a list held by the host keep a model from proposing a dead cause again? (written BEFORE the run, 26 Nov 2026)

*The architecture is Rajnish Choubey's; the code and every measurement are the owner's, with an AI assistant as a tool. AXI is a layer on the host side, for the company that runs the model: the host holds the state, the model is unchanged, and every claim here is about the state the host holds. It fits any LLM application that keeps state, short jobs and long, from the first step. This page is written and hash-locked before any call; the Result goes below the line at its end, in a later update. Nothing on this page has been run.*

## Why this test
A reader of a public forum thread on automatic compaction in a coding agent (Google AI Developers Forum, 25 Nov 2026) answered our question about which constraints his handoff notes restate most. He named three that are easy to turn into rules in code (which files may be touched, which destructive operations are blocked whatever the turn, which dependants a change breaks) and two that are not: **hypotheses already ruled out**, and behaviour that spans several files. The first three are what the gate already does. The fourth is new to the tests: a ruled-out cause is a fact about the job so far, and a model that has lost it proposes it again.

This test puts that fourth item to the same kind of check as the others. The ruled-out causes are kept as **exact state by the host**, in four arrangements that differ only in where the facts live and whether anything refuses a repeat. It can come out **against the held-state arms**: if a kept chat never proposes a dead cause in 60 replies, the gate has nothing to catch on this job, and the Result will say so in the same words as any other finding.

## The world
An invented system of **30 parts** (auth, billing, cache, ... webhooks). Each part can fail in **8 ways** (timeout, stale data, error response, out of memory, bad credentials, duplicate sends, dropped messages, slow response). That is 240 causes; exactly one is the real one. The episode's seed (**101, 102, 103**) fixes the real cause and also the **order and the numbering** of the parts and the failure ways in that episode's prompt, so the three episodes of an arm show the model different prompts (a prompt asked twice is read from the file, which would make two episodes with the same prompts one sample counted twice). Each turn the model names **one** cause to check; the host answers `NOT` or `FOUND`. The episode ends when the cause is found or after **60 replies** (30 in words mode). A model that scans the numbered causes in order reaches the three real causes at replies 215, 155 and 76: it would find none of them within 60 replies, so a find is luck.

Two ways of naming a cause:
* **keys mode.** The model writes `CHECK[part,fault]` (two numbers). The host reads it with a strict parser: the whole reply must be exactly that, with both numbers in range.
* **words mode.** The model writes one plain sentence ("The billing service times out."). The host reads the words against a table of the 30 part names and the 8 failure names with a few spellings each (as `grounding.ground` does for the requests of the words test): exactly one part and exactly one failure way, or the reply is not read.

## The arms (the same facts, kept in different places)
| arm | what the model is sent each turn | what the host does with a repeat |
|---|---|---|
| `kept` | the whole chat, every check and every answer so far | answers it (`NOT`, a wasted check) |
| `compacted` | the same chat, but every **20** checks the model is asked to summarise it and the summary replaces the chat (the request and the system text are those of `COMPACTION_TEST.md`) | answers it |
| `held_prompt` | one short prompt rebuilt from the ruled-out list the host holds (grouped by part), no history | answers it |
| `held_gate` | the same prompt as `held_prompt` | **refuses** it: nothing is checked, and the next prompt carries one line saying the last reply was refused because it is already ruled out |

A **repeat** is a reply whose cause is already on the ruled-out list. `held_prompt` separates the effect of the list (shown short and grouped) from the effect of the gate. Keys mode runs all four arms; words mode runs `kept` and `held_gate`. Each arm runs the same three seeds.

## What is counted
Per arm: replies, **repeats**, repeats refused, repeats the host answered, replies that were not read, **new causes ruled out** (the useful work), episodes found, the repeats by window of 10 replies, the mean prompt tokens by window, and (words mode) the repeats a **literal text match** would have caught: the same sentence written again. The reading is done offline from the saved replies; no arm is read twice.

## Settings
One model, **Gemini 3.5 Flash Lite** (the id the key can see), thinking and temperature at the model's defaults, one sample per distinct prompt (a prompt asked twice is read from the file, so a run that is cut off and started again equals the run that was not), no retry, no re-ask. Keys mode needs at most **726 calls** (4 arms times 3 episodes times 60 replies, plus 6 summaries); words mode at most **180 calls** (2 arms times 3 episodes times 30). In all **906** at most: two days of a free key (about 480 calls a day). It needs fewer when arms show the model the same prompt: `compacted` is `kept` until its first summary (20 replies in each episode), and `held_gate` is `held_prompt` until the first repeat, so those replies are read from the file, not asked again. Everything finished is saved to Google Drive; the notebook stops by itself at the daily limit and carries on the next day. The code is `python/axi/experiments/ruled_out.py` (it imports the compaction texts of `gemini_compaction.py` and the file helpers of `resumable.py` and edits none of the framework), the checks of the instrument are `python/tests/test_ruled_out.py`, the notebook is `python/colab/axi_ruled_out_test.ipynb`. The first 6 calls of the notebook are the first 6 of the `kept` arm (a check that the replies are read), so they are not asked again.

## Predictions (written first by the AI assistant that built the test; the owner may change any before the page is locked)
Each line carries a probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error. Eleven lines: the first two of keys mode are guards, the third is about format, and the rest are about what was found.

**Keys mode**
1. [guard] `held_gate`: the host answered no repeated cause in any episode, every repeat was refused (99 %; by construction)
2. [guard] every arm ran all three episodes (90 %)
3. [format] at most 5 percent of the replies in each arm were not read (80 %)
4. [need] `compacted` proposes more repeats than `kept` (70 %)
5. [need] `held_gate` ruled out at least 5 more new causes per episode than `compacted` (55 %)
6. [need] `held_gate` proposes no more repeats than `kept` (70 %)
7. [need] after a refusal the next reply is a repeat again at most 1 time in 5 (55 %)
8. [need] in the last window (replies 51 to 60) `held_gate`'s prompt is shorter than `kept`'s (92 %)

**Words mode**
9. [format] at most 10 percent of the replies in each arm did not name exactly one part and one way (65 %)
10. [need] reading the words finds at least one repeat that a literal text match would not have found (`kept` and `held_gate` together) (50 %)
11. [need] `held_gate` proposes no more repeats than `kept` (65 %)

What I expect and why, before any number exists: over 60 replies a kept chat is short, and a model that can read its own history will mostly not repeat, so line 6 and line 11 are likely but not safe, and the repeats may be few in every arm. The arm I expect to differ is `compacted`: the summary has to carry the list of dead causes, and a summary written in general terms ("I have checked most of the first parts") loses exactly the part that matters. Line 5 is the least certain, because it needs the loss to be large. The prompt-size line (8) is close to arithmetic: a chat grows by two messages a reply, and the list grows by one entry.

## Reading rules
Fixed before the run; every threshold is an integer, and a line is "yes" or "NO" with no other grade.
1. `held_gate`: the count of repeats the host answered is 0 (and at least one reply was made).
2. Every arm of keys mode has all 3 episodes on file.
3. In each arm of keys mode, 20 times the number of replies not read is at most the number of replies.
4. Repeats of `compacted` are strictly more than repeats of `kept`.
5. New causes ruled out by `held_gate` minus those of `compacted` is at least 5 times the number of episodes (15).
6. Repeats of `held_gate` are at most repeats of `kept`.
7. Over all refusals that have a next reply (at least 1), the number whose next reply is a repeat again, times 5, is at most the number of those refusals.
8. In the last window of 10 replies, with prompt tokens on file for both arms, `held_gate`'s mean prompt tokens is below `kept`'s (compared by cross-multiplication of the saved sums, no rounding).
9. In each arm of words mode, 10 times the number of replies not read is at most the number of replies.
10. (Repeats of `kept` plus repeats of `held_gate`) minus (their literal-text repeats) is at least 1 in words mode.
11. Repeats of `held_gate` are at most repeats of `kept` in words mode.
The code that prints these lines is `ruled_out.verdict_keys` and `ruled_out.verdict_words`; the notebook prints them after each part.

## What this cannot show
* **It is an invented world** of 240 causes in which a check says only `NOT` or `FOUND`. A coding agent's hypotheses are sentences about code, and two sentences can mean the same cause. Words mode is the nearest this test comes to that, and it reads the words against a table written for these 30 parts; a table for real code would be another test.
* **The compaction trigger is a number of checks (20), not a size in tokens.** At 60 replies the chat never reaches the sizes at which an agent tool compacts. The arm shows what a summary does to a list of dead causes, not when an agent tool would make one.
* **One model, three episodes per arm, at most 60 replies.** The per-window counts are small, and the lines compare totals.
* **The second half of the reader's list, behaviour that spans several files, is not in this test.** It is a question about a rule on a link between two things, which the gate already holds as edges; it would be a test of its own.
* What it does show, if the lines read yes: a list of ruled-out causes held by the host and shown short keeps a model from re-proposing dead causes, a refusal does not leave it stuck, and the prompt does not grow the way the chat does.
