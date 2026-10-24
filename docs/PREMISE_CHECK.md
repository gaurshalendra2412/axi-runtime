# The premise check: what a message takes for granted

*Concept-map item 128. Code: `python/axi/engine/premise.py`. Checked by test 154 (`python/tests/test_premise.py`). No model is called.*

## Where it comes from

Prompted by Rajnish Choubey's article "The Cinderella Project: Testing AI Understanding of Modern Visual Communication" (Medium, The Cinderella Project). In the AI chats it shows, the AI took each new premise it was given (a title, an age), then reversed it, and kept agreeing with whatever framing came last. A chat with no held facts lets the latest message win.

## What is new

AXI already keeps the facts with the host, and a question can never change them. Its check before a change asks whether the change is **valid**: the item is on the list, the value is allowed. Nothing asked whether what a message **takes for granted** is **true**.

`premises` reads a message with one word, one reading (item 127). In every sentence that names exactly one item the host holds, it ties each value the sentence carries (a day, an amount, a status, an owner) to that item, and compares:

* a value marked as the old one ("not Monday", "from Monday", "instead of Monday") with the list before the step;
* every other value with the list the host would commit after the step (for a question, the list does not change).

A value the list does not hold is reported: "Note: you said Friday for the day of Amber Gallery; the list has Monday." The report changes nothing: the check, the repair and the commit run as before. A sentence that names no item, or two, ties nothing, so the chatter around a message cannot raise an alarm.

## What it gives (offline)

**No false alarm on Test 23.** All 400 steps, each message with its 30 sentences of office chatter (full of days, amounts, owners and statuses), against the list before the step and the list after AXI's reply was committed: **0 reports**, while 214 values in the messages were tied to an item and checked.

On false premises **written for this check** from Test 23's stream, each also written with the true value:

| Messages | False premise reported | True premise reported |
|---|---|---|
| 29 questions ("Since Dune Gallery is on Wednesday, what is the amount of Dune Gallery?") | **29 of 29** | 0 of 29 |
| 100 change messages ("Dune Gallery is on Wednesday, so Hand Dune Gallery to Marcus.") | **100 of 100** | 0 of 100 |

The host still answers 21 of 29 of these questions itself, all exactly; the other 8 (a premise about the owner) go to the model with the note. The check takes about 1.5 ms a step with the chatter and 0.1 ms for the message alone; a model call took 2.76 s a step in that run.

## What this does not show

What the check cannot see: a reply that writes the false premise into the list as well ("Dune Gallery is on Wednesday, so ..." answered by also setting the day to Wednesday). The list after the step then holds it, and none of those 100 was reported. A plain statement about a held item can be a premise or an update, and the words alone do not say which. Also not shown: how real messages are worded; the false premises are ours.
