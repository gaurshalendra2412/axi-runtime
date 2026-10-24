# R = 0 for questions: the host answers what it holds

*Concept-map item 126. Code: `python/axi/engine/answer.py`. Checked by test 152 (`python/tests/test_answer.py`). No model is called; the figures are counted offline from Test 23's stream and saved calls.*

## Where it comes from

Prompted by Rajnish Choubey's open letter "The Failure of Thought Imposition" and its Spherical Mind. The radius R is everything a machine is made to engage. At R = 0 "the question is the answer" and "how you process: silence": the output is minimal and carries the most meaning. At R = Max everything is engaged and the output is exhaustive, made for show. The letter also asks to "abolish the Jargon Filter: plain, simple truth is exponentially more intelligent than complex, redundant analysis." The method that carries the idea is a plain lookup in the exact state.

## What is new

Until now every question about the state went to the model, which read the whole list and wrote the answer. But the host already holds the exact answer. `answer` reads the question for the one field it asks about and for the item, by name or by a description resolved against the held facts (item 122), and then:

* one item, the fact held: the value;
* one item, the fact never given: "unknown";
* a description that fits two or more items: "ambiguous";
* anything it cannot place (no field, two fields, two items, a description that fits nothing): the model is asked, as before.

No model call for the questions the host can place, and the host never invents a value. Since item 127 (`ONE_READING.md`) it also reads every word of the question once, so a word inside a name ("Day" in "Day Spa") is never read as a field too.

## What it gives on Test 23 (offline)

Of Test 23's 400 steps, 39 were questions. The host answered **39 of 39**, every one exactly: 37 by name (5 of them "unknown", a fact never given) and 2 "ambiguous". The model had answered all 39 exactly with AXI (38 in the kept chat), so the gain is not in exactness on this run; it is in cost and certainty. Those 39 questions used **9.2 %** of AXI's computed tokens and **7.7 %** of its seconds in that run; answered by the host, they cost neither.

## What this does not show

How real questions are worded: Test 23's questions follow the stream's templates, which the host reads by names, field words and descriptions. A question in another form goes to the model, as before.
