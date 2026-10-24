# One word, one reading

*Concept-map item 127. Code: `read` in `python/axi/engine/answer.py`. Checked by test 153 (`python/tests/test_one_reading.py`). No model is called.*

## Where it comes from

Prompted by the text on Devanagari and the Latin alphabet from Rajnish Choubey's conversation (the owner pasted it on 24 Oct). Devanagari has one sound for one symbol, and every consonant has one place in its table, set by where in the mouth it is made (the row क ख ग घ ङ is made at one place and differs only in voicing, breath and the nose). English has "their / there / they're", one sound with several spellings, and "read / read", one spelling with several sounds. A reader that must answer exactly needs the first kind of code.

## What it fixes

The host's answers (item 126) read a question three times on their own: once for the field word, once for the item name, once for a describing value such as an owner. So one word could be read twice. With an item called "Day Spa", "how much is Day Spa?" read "Day" as part of the name and also as the field "day", and the host answered "Monday", a wrong answer to a question about the amount. "What is the amount of Day Spa?" found two fields and went to the model for nothing; "what is the day of Priya Hall?" read Priya as an owner and went to the model too.

`read` looks for every phrase of every kind at once, longest first. A span taken by one phrase is never read again inside another, so "Day Spa" is the name and its "Day" is nothing else. A phrase that belongs to two kinds (an item named "Priya" when Priya is also an owner) is not given a reading: the question has two readings and goes to the model.

## What it gives (offline)

**No change on Test 23**: its 39 questions get the same answers as before, 39 of 39 exact, and the 32 recall questions reworded without the field word ("how much is ...?") are all handed to the model, as before.

On the same questions with names that hold field words or owners' names (**written for this check**: the first word of every name replaced by Day, Status, Amount, Owner or an owner's first name):

| Reading | Questions answered by the host | wrong | Reworded questions answered wrong |
|---|---|---|---|
| Each kind on its own (Update 86) | 6 of 39 | 0 | 13 of 32 |
| One word, one reading | **39 of 39** | **0** | **0** (all 32 go to the model) |

The reading takes far under a millisecond a question.

## What this does not show

How real names and questions are worded: the renamed stream is ours. A question the host cannot place still goes to the model, as before.
