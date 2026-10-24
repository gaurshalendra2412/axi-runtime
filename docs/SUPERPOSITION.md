# Hold every reading, choose none

*Concept-map item 125. Code: `python/axi/engine/superpose.py`; the comparison stream: `python/axi/experiments/ambiguity_desk.py`. Checked by test 151 (`python/tests/test_superpose.py`). No model is called.*

## Where it comes from

Prompted by Rajnish Choubey's essay "The Wave, the Unicorn, and the End of Western Psychology". Light is a particle when observed and a wave when not; observation only decides what is seen. "How can one select one? That would be foolish." The method that carries the idea is a standard one: the host keeps a small set of possible states (possible worlds).

Read for structure, two parts of the essay are already in AXI: "replace every OR with AND" is the gate, whose four conditions must all hold, with no ranking among them (a shuffled change never changes the verdict); and "observation decides what we see" is the whole exact state held by the host against the view shown to the model (item 124). One part is new.

## What is new

Until now a change that fits more than one item ("whatever Priya is handling: the day is now Sunday" when Priya has two) applied nothing. That is safe, but the change is lost. With the readings layer the host keeps every reading instead:

* a named change is made in every world; a world where it cannot apply drops out;
* an ambiguous change splits every world into one world per item it could mean, plus one where it applied nowhere (today's rule);
* a later message that says which item was meant keeps only the worlds that read it that way;
* a question is answered only when every world gives the same answer; otherwise it is not settled yet, never guessed;
* a reading not clarified within 8 steps falls back to today's rule, and above 9 worlds the host applies nothing, as today.

The check before each change runs in every world, on that world's exact state. With one world, every call is the plain change.

## No harm where there is no ambiguity

Test 23's saved replies, run through the readings, give exactly the same list as before at **400 of 400** steps, with **one world** throughout. The host's own work is about 0.14 ms a step against 0.10 ms on the plain path, on one CPU; a model call took 2.76 s a step in that run.

## What it gives where there is ambiguity

On a stream **written for this comparison** (5 seeds of 400 steps; one step in twenty is an ambiguous change, 94 in all, 76 of them clarified within a few steps; 517 questions; every message assumed read correctly, so only the host's rule differs):

| Host rule | Steps with the exact list | Questions answered right | answered wrong | not settled yet |
|---|---|---|---|---|
| Today: apply nothing | 314 | 490 | 27 | 0 |
| Guess the first item that fits | 350 | 482 | 35 | 0 |
| Hold every reading | **827** | **503** | **6** | 8 |

Holding every reading had the exact list on 519 steps where today's rule did not, and missed it on 6 steps where today's rule had it (each while a reading was still waiting for its clarification). 18 readings were never clarified and fell back to today's rule; 10 times more than 9 worlds would have been needed and the host applied nothing, as today. The true list was among the worlds held on 995 of the 2,000 steps.

The absolute numbers are low for every rule because one change in twenty is ambiguous and a lost change stays wrong until it is overwritten; the comparison is between the rules on the same stream.

## What this does not show

How often real requests are ambiguous, or that a model writes a description ("whatever Priya is handling") instead of guessing a name. The stream is ours, written for this comparison, not a real workload.
