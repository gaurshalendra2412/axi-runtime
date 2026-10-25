# Receive without adding: the host applies a plain change itself

*Concept-map item 129. Code: `python/axi/engine/receive.py`. Checked by test 155 (`python/tests/test_receive.py`). No model is called; every figure below is counted offline from the Test 23 stream and its saved calls.*

## Where it comes from

Rajnish Choubey's article "a durga diary: Why do we park in driveways and drive on parkways?" (Medium, 28 May). The article is a long chat with Claude and Gemini side by side, and his lines appear inside their replies, so the exact wording may be the AI's paraphrase of what he said. The lines read for this item:

* the user's input is a direct experience, complete, and needs no addition: "the addition is the failure";
* a tool that "cannot receive without immediately responding";
* an architecture that "doesn't know how to stop".

The same line runs through two more of his articles: "What Exactly Will AI do?" (a good loudspeaker passes the sound on without adding anything of its own) and "SAARE AI ke bacho YAHAN AAO" ("the pulse should stay raw"). Both are also chats, with the same caution about wording.

Read for structure, most of this is already AXI. A fact never given is "unknown", never a guess (item 126). Knowing when to stop is the host answering a question itself, with no model call (item 126). Checking a claim against the list, not against another model's word, is the premise check (item 128).

## What is new

Item 126 took the model off the questions the host can answer from the list. This item takes the model off the **changes** the host can read for itself.

When a message is a plain change ("Hand Dune Gallery to Marcus."), the host builds the change and sends it to **the gate**, the same check every model reply passes. If the gate admits it, the change is committed. No model is called and no reply text is written: the message is received, and nothing is added.

The host reads the message with one word, one reading (item 127). It acts only when the message is plain:

* exactly one sentence names a held item, and it names exactly one. A sentence that names none, such as office chatter, is not read;
* that sentence asks nothing ("?");
* it gives no reason ("so", "since", "because", "given", "therefore"). A reason is a premise, not a change (item 128). "As" is not read as a reason, because it also marks a role: "Mark Amber Gallery as confirmed";
* it adds no aside ("which", "who", brackets). An aside is a premise too;
* it hedges nothing ("if", "maybe", "might", "could", "should", "perhaps", "probably", "unless", "when", "think");
* it negates nothing. "Not" is allowed only as the mark of an old value, as in "Marcus, not Elena";
* it carries one new value for each field it changes, and each value differs from the list. An old value marked "not", "from", "instead of" or "rather than" is left out. With no value at all, it carries a word of removal (drop, remove, delete, cancel, scrap, forget, off).

Everything else goes to the model as before, and so does anything the gate refuses. The host never adds an item: a new name is not on the list it can read. It never reads a loose reference ("the last one we added", "the one Priya is handling", "the same day as ..."), because those name no item or two. It edits no framework file.

## What it gives on Test 23 (offline)

Test 23 (seed 20261122) has 400 steps: 39 questions and 361 changes.

| | Changes |
|---|---|
| Applied by the host and exact | **199 of 361** (102 set, 41 delete, 32 set two fields, 24 corrections) |
| Applied by the host and wrong | 0 |
| Handed to the model: no held item named (67 adds, 48 "the last one", 23 "the one Priya is handling") | 138 |
| Handed to the model: the sentence names two items ("the same day as ...") | 24 |

The result is the same when each message sits inside its 30 sentences of office chatter.

When AXI's arm A ran Test 23, the model's calls on those 199 steps took **510.5** of its **1,102.8** seconds (46.3 %) and **261,093** of its **528,836** computed tokens (49.4 %). The host takes those steps off the model.

Add the 39 questions the host answers itself (item 126; 85.4 seconds and 48,858 tokens). Then the model would be needed on 162 of the 400 steps, and 54.0 % of the seconds and 58.6 % of the computed tokens would not be spent.

**Further checks:**

* On 20 more seeds of the same stream, 4,212 of 7,220 changes were applied, none wrong.
* With names that hold field words and owners' names (item 127's renaming, such as "Day Gallery" or "Priya Studio"), 199 were applied, none wrong.

**Speed.** The host's reading takes about 1.7 ms a step with the chatter and 0.1 ms for the message alone. It slows nothing down: a step the host takes is a model call that is not made.

## On messages written for this check

From Test 23's set steps, each message was rewritten so that it carries something other than a plain change, about 100 messages of each form. None of them was applied:

* a false premise given as a reason ("Dune Gallery is on Wednesday, so Hand Dune Gallery to Marcus.", and with "since" or "because");
* a false premise as a sentence of its own, or as an aside ("which means ...", brackets);
* a negation ("Don't ...", "Do not ...");
* a hedge ("Maybe ...", "If the client agrees, ...", "I think ...");
* a question ("... ?", "Should we ... ?");
* a value the list already has.

What it **cannot see**: a statement joined to the change by a plain "and" ("Dune Gallery is on Wednesday and hand Dune Gallery to Marcus."). All 100 of those were applied, and the stated value was written too, as a second change. A bare statement of a new value ("Amber Gallery is on Friday.") is also taken as a change, as the model and Test 23's ideal take it. If either was meant as a premise, the premise check (item 128) cannot see it either.

## What this does not show

* How real messages are worded: the measured messages follow Test 23's templates, and the rewritten ones are ours.
* That a user would not want a reply. The host commits the change and says nothing of its own; what the user sees afterwards is the list.
* The plain-"and" case above.
