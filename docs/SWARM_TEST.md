# Can many agents edit one shared state at the same time? A concurrency test of the gate and the pager (written BEFORE the run, 15 Oct 2026)

This page fixes the question, the rule being tested, the measures, the predictions and the reading rules before any number from this test exists. Code: `python/axi/experiments/swarm.py`; instrument checks `python/tests/test_swarm.py`. Concept-map item 100. It needs no GPU, no model and no internet: it runs on a CPU (Codespaces is enough) in about two minutes.

## Where the idea comes from, and how it is treated
On 15 Oct 2026 the owner pasted a text (author not stated to me; it speaks of "Choubey's Mapping (axi-cache)" and I have not checked with Rajnish Choubey that the formulations are his, so I do not attribute them to him). One part says that, because the framework "proves that conversational history is irrelevant", an agent is "completely stateless", so that "10,000 AI agents can read and write to the exact same enterprise state graph simultaneously using Copy-on-Write (CoW) shadow paging", that "Without AXI, multi-agent swarms inevitably overwrite each other's context and corrupt the shared environment", and that "AXI acts as the atomic transaction manager for the swarm". No test or number is given.

I first answered by asking whether the repository already had it. The owner objected: a thing that is missing from the repository is a **new** thing, and not being there is no reason to set it aside. This page takes that seriously: the claim is treated as a hypothesis worth a measurement. The two other claims of the same text (a switch to packed bit matrices for dense graphs, and a snap of the model's last vector onto a lattice of valid commands) are different questions; the first is item 101, the second is not tested yet.

What exists today, as far as I read it (`engine/gate.py`, `engine/pager.py`, `engine/runtime.py`, `engine/dist_commit.py`, `crates/axi-cache`, `docs/RFC-001.md`): the gate judges ONE edit against ONE state; the pager is a copy-on-write page table whose commit finds a `Conflict` when a page it worked on has been changed by someone else; `Runtime.propose` handles one proposal at a time, start to finish, so two proposals made on the same snapshot and open at the same moment are never exercised; `DistributedPager` serializes proposals through a coordinator. I searched the repository for "concurrent", "simultaneous" and "swarm" and found no test of this. **So the question below has not been asked of this code.**

## What is tested, and what is not
**Tested (CPU, no model):** (1) which rule tells correctly whether two edits made on the same snapshot clash; (2) what happens in a round where N agents make edits on the same snapshot, under three ways of handling them; (3) how the share of edits that can be taken together falls as the number of agents grows and rises with the size of the graph; (4) a store shared by eight real threads.

**Not tested here, and not testable without a model or a cluster:** that an LLM agent needs only the current state (the claim of statelessness); what an agent proposes; 10,000 agents (the largest round here has 64, the threaded runs have 8); processes on several machines; the Rust crate (the Python `Pager` is a model of its semantics, not the crate); the RFC's "O(1) atomic pointer swap" (the Python model says commit is proportional to the pages touched; neither is measured here).

## The rule that is tested
It is a classical result of graph rewriting (parallel independence of double-pushout rules, the Church-Rosser property; see Related work): two rewrites made on the same graph can be done in either order, with the same result, when what they both **match** is only what they both **keep**. For the edits of the gate (add or delete a service, add or delete an edge), with the items written `("n", service)` and `("e", (from, to, relation))`:

* **L**, what an edit matches: the items it deletes, and the items it needs to exist, which are the two ends of every edge it deletes or adds (an end that the same edit adds is not needed from outside).
* **K**, what it keeps: L without the items it deletes.
* **C**, what it creates: the services and edges it adds.
* `independent(a, b)` holds when (L_a ∩ L_b) is inside (K_a ∩ K_b), and C_a ∩ C_b, C_a ∩ L_b and C_b ∩ L_a are all empty. The last condition is needed because the gate names its items (there are no fresh identities), so two agents can make the same service.

Examples: two edits that each add an edge from the same hub service are independent (the hub is matched and kept by both). One edit that deletes a service and another that adds an edge to it clash (the service is matched by both and deleted by one). Two edits that delete the same edge clash (the edge is matched by both and deleted by both). Two edits that add the same service clash (C overlaps).

## Three detectors and the truth
Each detector sees two edits made on the same snapshot and says "clash" or "fine":

| detector | says "clash" when |
|---|---|
| `dpo` | `independent` is false |
| `pager` | the real `Pager` raises `Conflict`: two transactions are open on the same table, each doing what `Runtime.propose` does for its edit (write the page of every added service and of the ends of every added edge, unmap the page of every deleted service); one commits, then the other |
| `names` | the items the two edits write (create or delete) overlap |

**The truth:** apply the two edits to the snapshot in both orders with the real gate (`gate.check`, `gate.apply`). They **commute** when both orders are accepted and give the same graph; otherwise they do not. A detector **misses** a pair that does not commute and calls "fine"; it gives a **false alarm** on a pair that commutes and calls "clash".

## The experiments
**Pairs.** 20 graphs per regime, each of 40 services and 60 edges on a 10 x 10 grid (the generator of the large test, `kv_large.random_large_graph`), and 6,000 pairs of edits per graph: 120,000 pairs per regime, 240,000 in all. Each edit is one of six kinds, drawn with these weights: add a service 15, add an edge 30, delete an edge 20, delete a service without edges 10, delete a service with its edges (1 to 4) 15, and a compound edit 10 (a new service wired to an existing one, or one edge replaced by another); the gate must accept it on the snapshot or it is drawn again. Two regimes: `uniform` (every choice at random) and `hot` (half of the choices go to the three services with most edges, the three first free cells, or the edges at those services: a crude picture of a crowd of agents working on the same few things). Seed 41.

**Rounds.** In a round, 16 agents each make an edit on the same snapshot (the same graphs and kinds, 100 rounds per regime); the edits arrive in 10 random orders. Three ways to handle them: `blind` (apply every edit, no gate: an add writes its key over whatever is there, a delete removes it if it is there); `gated` (first come, first served: each edit goes through the gate against the state as it is when its turn comes); `merge` (go through the agents in the order of their number, not of arrival, and keep an edit if it is independent of all the ones already kept; then apply the kept edits in the order of arrival, each through the gate).

**Sweep.** The share of the agents whose edit the merge keeps, for 4, 16 and 64 agents on graphs of 40, 160 and 640 services (60, 240 and 960 edges), no hot spot, 400 rounds each.

**Threads.** Eight real threads, 300 rounds. In a round each thread takes a snapshot, all wait for each other (so that all eight hold the same snapshot), each makes an edit from it and commits, racing for the lock; after even rounds all wait again, after odd rounds they do not, so that some snapshots are stale. Graph of 60 services and 90 edges, `hot` regime. Three stores: `rule` (an edit is committed if it is independent of every edit committed after its snapshot AND the gate accepts it on the current state), `gate` (the gate on the current state only) and, as a control for the instrument, `lww` (the edit is applied to the agent's own snapshot and that becomes the state: last writer wins the whole state). After each run the log of accepted edits is replayed from the start through the gate and compared with the final state.

**Debugging before the page was locked:** the code was run on batches of at most 400 pairs, 5 rounds and 20 thread rounds to see that it ran and how long it took; I did not print or record any rate. My estimates below were made by hand from the edit mix.

## Predictions (mine, written first)
Each line carries my probability that it reads "yes". They are the reading rules below, in the same order. A line that comes out "NO" is a finding, not an error.

1. [instrument] Enough pairs of both kinds: over the two regimes at least 2,000 pairs do not commute and at least 20,000 do. (92 %)
2. [instrument] In every pair both edits pass the gate on the shared snapshot, and the first commit of the two pager transactions succeeds (so a Conflict can come only from the second). (99 %)
3. [instrument] Every thread of the threaded runs finished every round (no worker raised, no barrier broke), for all three stores. (97 %)
4. [instrument] The control store that lets the last writer replace the whole state loses updates: its log replayed through the gate does not give its final state. (99 %)
5. [claim] dpo: no pair that does not commute is called independent (0 missed), in both regimes. (90 %)
6. [claim] dpo: at most 1 % of the pairs that commute are called a clash (false alarms), in both regimes. (88 %)
7. [claim] pager: at least 10 % of the pairs that do not commute raise no Conflict (missed), over the two regimes. (90 %)
8. [claim] pager: at least 2 % of the pairs that commute raise a Conflict (false alarms), in the hot regime. (70 %)
9. [claim] names: at least 5 % of the pairs that do not commute have no written item in common (missed), over the two regimes. (93 %)
10. [claim] merge: in every round of the order test (both regimes) the chosen set gives the same final graph in all 10 arrival orders, and every step of it passes the gate. (97 %)
11. [claim] gated (first come, first served): in the hot regime at least 80 % of the rounds end in different graphs for different arrival orders. (80 %)
12. [claim] blind (no gate): in the hot regime at least 10 % of the rounds end in a graph with an edge whose end is missing, for at least one arrival order. (92 %)
13. [claim] gated and merge: no round ends in a graph with an edge whose end is missing. (99 %)
14. [claim] merge: the share of agents whose edit is chosen falls as the number of agents grows (4, 16, 64) for every graph size, and rises with the graph size (40, 160, 640 services) for every number of agents (no rise or fall against the trend by more than half a point, and at least a point from end to end). (90 %)
15. [claim] merge: with 64 agents the share chosen is between 45 % and 80 % on the graph of 40 services and at least 90 % on the graph of 640 services. (75 %)
16. [claim] threaded store with the independence check: the log replayed through the gate gives the final state, the state is well formed, every attempt is accounted for, and the gate never refuses an edit that the check passed. (95 %)
17. [claim] threaded store with the gate alone (no independence check): the same four properties hold. (93 %)
18. [claim] threaded store with the independence check: at least 3 % of the attempts are refused as clashes (the clash path was exercised). (90 %)
19. [claim] gated and merge take about the same share of the edits: within 0.03 of each other in both regimes. (95 %)

**Headline** (lines 5, 6, 10 and 16): if all four read "yes", then for the edit language of the gate the independence rule is exact (it calls "clash" exactly when the two edits really do not commute), a batch handled with it ends in the same state whatever the order of arrival, and a store shared by real threads stays correct. The product of my four probabilities is about 73 %; they are positively correlated (if the rule is right, the other three follow from it), and my single number is about **70 %**.

Why I think so, line by line. *Lines 5 and 6 (90 %, 88 %):* the rule is the textbook condition plus one added clause for named items; in this language I could not build a case where it fails (I tried deleting an edge that the other deletes with its service, an edge added to a service that the other deletes, two adds of one name, an edge added to a service the same edit adds); what could break it is something I did not think of, hence 90 and not 99. I also expect the rule to be exact, which is stronger than safe: I think every clash in this language has an order in which the second edit is refused by the gate. *Line 7 (90 %):* two agents deleting the same edge, or one deleting an edge and the other the service that carries it, touch no page in `Runtime` (an edge delete writes no page and a service delete unmaps only the service's page), so the pager cannot see it; I estimate these at one clash in four. *Line 8 (70 %):* two agents adding an edge from the same hub both write the hub's page, and the second commit fails, although the edits are independent; by hand I get about 3 % of the commuting pairs in the hot regime, close enough to the 2 % line that I give it 70. *Line 9 (93 %):* an edit that adds an edge to a service and another that deletes the service share no written item, which I estimate at four clashes in ten. *Line 11 (80 %):* with about 3 clashes per round of 16, most rounds have a clash whose winner depends on arrival; some clashes (two deletes of the same edge) end in the same graph whichever wins, which is why this is not 99. *Line 12 (92 %):* about one pair per round is an add-edge-to-a-deleted-service, which the blind scheduler cannot refuse. *Line 14 and 15 (90 %, 75 %):* if a pair clashes with probability c, about 2 % here for 40 services, then of 64 agents a greedy choice keeps about 65 % (a fixed-point estimate), and c shrinks roughly in proportion to the number of services, so the 640-service graph keeps over 95 %; my estimate of c may be 30 % off in either direction, hence the width of the band. *Line 18 (90 %):* about 0.7 clashes per round of eight, so about 8 % of the attempts. *Line 19 (95 %):* both pick a maximal set greedily in a random order of the same agents.

What each outcome would mean, written now so that I do not read it afterwards to suit the numbers. *Lines 5 and 6 yes:* for this language the footprint check is a complete, cheap test of whether two agents' edits can be taken together, and it can be run without the current state (on the two edits alone), which a gate on the head cannot; a shard or a replica could use it. *Line 5 NO:* the rule has a hole, and the first example the run prints is the case to study; a store that relied on it would have needed the gate as a second check (line 16 counts the cases where the gate refuses what the rule passed). *Line 6 NO but 5 yes:* the rule is safe but loses throughput; the false alarms say which kinds. *Lines 7 and 8 yes:* the pager as it is cannot be the judge of a swarm: it misses real clashes (so a gate on the current state at commit time is needed, which `Runtime` does only because it serializes) and it raises false ones on hubs (a hub service would be a bottleneck); the remedy is to run the rule before the pager, or to give the pager finer pages (one per edge). That is a finding against the text's wording "CoW shadow paging acts as the transaction manager": shadow paging alone does not decide what may be taken together; the gate and the rule do. *Line 17 yes:* with one coordinator that holds the current state, the gate alone already keeps the state correct, and the independence rule adds order-independent batching and checks without the head, not safety. *Lines 11 and 12 yes:* the strawmen fail as expected, which shows the measures can tell, not that anything is proved about AXI. *Lines 14 and 15:* they give the number that decides whether "10,000 agents" is a reasonable thing to say: the share of edits that can be taken in one round falls with the number of agents divided by the number of services; where it stays high, a graph has to be much larger than the crowd.

## Reading rules (fixed now; the printout prints exactly these)
The nineteen lines above are the reading rules; `swarm.verdict_swarm` computes them from the saved result. A pair is counted once, as an ordered pair of two edits drawn independently from the same snapshot (the two may be the same edit, which is a clash). "Missed" and "false alarm" are defined above, against the truth of applying both orders through the real gate. In a round, "the same final graph" means equal sets of services (with values) and edges (with weights). "An edge whose end is missing" is an edge in the final state whose from-service or to-service is not in it. In the sweep the share chosen is the number of kept edits over the number of agents, averaged over rounds. In the threaded runs, "accounted for" means that the attempts equal the accepted plus the refused, and that the length of the log, the version of the state and the number accepted are the same number. The thread runs are not repeatable in their counts (the order of commits is the scheduler's), only in what must hold.

## What this cannot show
Random edits, not an agent's: the mix of kinds, the hot regime and the sizes are mine, every rate depends on them, and a real crowd works on what its task is about. One process with eight Python threads (the interpreter lock serializes bytecode, so this tests the logic of the check-and-commit under interleavings, not hardware races), no processes, no network. The edit language has no in-place update (`SET`, concept-map row 17); an update would add a read-write kind of clash that this rule would have to be extended for. The rule decides whether two edits can be taken **together**, not whether the result is what the two agents **meant**: two agents who each delete a different service to "improve" the same metric are independent here. The pager is the Python model, not the Rust crate. Statelessness of the agents, the quality of what an LLM proposes, retries after a refusal, and anything at 10,000 agents are not covered.

## Related work (from memory, not searched; check before citing)
Parallel independence and the Church-Rosser property of double-pushout rewriting: Ehrig, Pfender and Schneider (1973), and the survey of Corradini et al. in the *Handbook of Graph Grammars and Computing by Graph Transformation*, vol. 1 (1997). Optimistic concurrency control, where a transaction validates its reads and writes against those committed since it started: Kung and Robinson (1981). Snapshot isolation and its anomalies: Berenson et al. (1995). Conflict-free replicated data types: Shapiro et al. (2011). If these hold as I remember them, the finding that independent footprints commute would not be new in the theory; what would be new here is the measured exactness of the rule for this edit language, the measured gap between page-level and rule-level clash detection in this pager, and the throughput curve.

## How it is run
In Codespaces, in `python/`: `python -m axi.experiments.swarm --out axi_swarm.json` (about two minutes, CPU). The pairs, the rounds and the sweep are repeatable from the seed: the printout shows a digest of them, which should be the same on any machine with the same code. The result file is JSON; send it back and this page gets a "Result" section without changing anything above this line.


---

# Result

*Written after the run, below the locked text; nothing above this line has been changed (its hash is checked by `tests/test_swarm.py`). The run was made on 15 Oct 2026 in my workspace, not on the owner's machine: `python -m axi.experiments.swarm --out axi_swarm.json`, seed 41, 82 seconds on two CPUs with Python 3.13. The result file is `docs/results/axi_swarm_seed41.json` (digest of pairs, rounds and sweep: **dd08fd515ff59bff**). The same run under Python 3.11 gives the same digest and the same pairs, rounds and sweep (`axi_swarm_seed41_python311.json`); only the counts of the threaded runs differ, as the page said they would. Test 109 (`tests/test_swarm_result.py`) recomputes every figure below from the saved file.*

## The 19 lines

| # | my probability | what was measured | read |
|---|---|---|---|
| 1 | 92 % | 4,842 of the 240,000 pairs do not commute (2,369 uniform, 2,473 hot), 235,158 commute | yes |
| 2 | 99 % | both edits pass the gate in all 240,000 pairs; the first commit of the pager succeeds in all | yes |
| 3 | 97 % | all threads finished all rounds, for all three stores (no error, no broken barrier) | yes |
| 4 | 99 % | the last-writer control: 2,400 of 2,400 attempts "accepted", and its log replayed through the gate does not give its final state | yes |
| 5 | 90 % | `dpo` missed **0** of 2,369 and **0** of 2,473 | yes |
| 6 | 88 % | `dpo` false alarms **0** of 117,631 and **0** of 117,527 (0 %) | yes |
| 7 | 90 % | `pager` missed 651 + 669 = 1,320 of 4,842 clashes: **27.3 %** (27.5 % uniform, 27.1 % hot) | yes |
| 8 | 70 % | `pager` false alarms in the hot regime 2,901 of 117,527: **2.47 %** (the line was 2 %) | yes |
| 9 | 93 % | `names` missed 1,123 + 814 = 1,937 of 4,842: **40.0 %** (47.4 % uniform, 32.9 % hot) | yes |
| 10 | 97 % | `merge`: one final graph for all 10 arrival orders in **100 of 100** rounds in both regimes; the gate refused nothing after the rule (0) | yes |
| 11 | 80 % | `gated`: different graphs for different arrival orders in 83 of 100 hot rounds (81 uniform); the line was 80 | yes (by 3 rounds) |
| 12 | 92 % | `blind`: an edge with a missing end in 52 of 100 hot rounds (68 uniform) | yes |
| 13 | 99 % | `gated` and `merge`: 0 ill-formed rounds in 400 | yes |
| 14 | 90 % | the share chosen falls with the number of agents and rises with the size of the graph in all nine cells | yes |
| 15 | 75 % | 64 agents: 69.9 % on 40 services (band 45 to 80 %), 96.2 % on 640 (line: at least 90 %) | yes |
| 16 | 95 % | threads, rule store: replay equals the final state, well formed, 2,400 attempts accounted for (2,083 accepted + 317 refused as clashes), 0 gate refusals after the rule | yes |
| 17 | 93 % | threads, gate-only store: the same four properties (2,035 accepted, 365 refused by the gate) | yes |
| 18 | 90 % | clash refusals 317 of 2,400 attempts: 13.2 % (line: 3 %) | yes |
| 19 | 95 % | `gated` took 0.874 and `merge` 0.875 of the edits (uniform), 0.874 and 0.873 (hot): a gap of 0.001 | yes |

**Headline (lines 5, 6, 10, 16): all four read yes.** My probability for the four together was 70 %. For this edit language, on these random edits, the independence rule is exact (a clash exactly when the two edits really do not commute), a batch handled with it ends in the same state whatever the order of arrival, and a store shared by eight real threads stayed correct.

**All 19 predictions came out as I wrote them, and I do not take that as a sign that the test is strong.** I wrote the instrument, the truth and the predictions; the rule is a textbook one; and several of my "yes" had a margin of only a few points (line 8: 0.47 points; line 11: 3 rounds). The check against sampling is the exhaustive pass below.

## Where my hand numbers were off
The page asked to say it. The share of pairs that clash I put at 2 to 2.5 %: it is 1.97 % and 2.06 %. The misses of `names` I put at about 40 %: 40.0 % over both regimes, but 47.4 % in the uniform regime and 32.9 % in the hot one, a spread I had not expected. The false alarms of `pager` in the hot regime I put at about 3 %: 2.47 %; and **I did not expect any in the uniform regime**: there are 1.36 % (1,594 pairs), of which 1,028 are two plain edge additions that share an end. The `pager` gives one page to each service (`vpn_of`), and `Runtime.propose` writes the page of both ends of every added edge, so two agents who each attach an edge to the same service collide on a page that neither changes in substance. The clash share in the threaded runs I put at about 8 %: 13.2 %. The blind scheduler's ill-formed rounds I bet at 10 % or more and they are 52 to 68 %; and the hot regime has fewer than the uniform one (52 against 68), which I did not expect and cannot explain from this data. The share kept at 64 agents on 40 services: my fixed-point estimate was 65 %, it is 69.9 %.

## What I read from it (derived afterwards; none of this was pre-registered)
1. **`merge` does not keep more edits than first come, first served; it makes the outcome independent of the order.** Both take 87.3 to 87.5 % of 16 edits (line 19). What differs is that `merge` ends in one graph in 200 of 200 rounds, whatever the order of arrival, and first come, first served in 36 of 200.
2. **How much of a batch can be taken together depends on agents divided by services.** Share chosen by the merge, by agents per service: 0.006 (4 on 640): 99.8 %; 0.025 (4 on 160 and 16 on 640): 99.0 % and 99.1 %; 0.1 (4 on 40, 16 on 160, 64 on 640): 97.6 %, 96.2 % and 96.2 %; 0.4 (16 on 40, 64 on 160): 88.1 % and 87.2 %; 1.6 (64 on 40): 69.9 %. The nine points lie on one curve. If that curve held further out, which this test cannot show, 10,000 agents would keep about 96 % on a graph of about 100,000 services and about 70 % on one of about 6,000.
3. **The `pager` as it is written is wrong in both directions, as a judge of what may be taken together.** It misses 27.3 % of the real clashes: the cases are two agents deleting the same edge, an agent deleting an edge and another deleting the service that carries it, and two deleting neighbouring services (a delete of an edge writes no page). And it refuses independent pairs, mostly two agents attaching an edge to the same service. This is a statement about the Python `Pager` and about what `Runtime.propose` does with it, not about the Rust crate.
4. **With one head, the gate alone already keeps the state correct** (line 17: 2,400 attempts from eight threads, 365 refused by the gate, replay equal, state well formed). What the rule adds is (a) a deterministic batch, (b) a check that needs only the two edits and not the current state (a replica or a shard could run it), and (c) a measure of how much a crowd can do at once (item 2 above). It does not add safety to a gate that sees the head.
5. **Every one of the 4,842 clashes was a pair for which the gate refuses both orders** (`both orders refused`); no pair on a shared snapshot was accepted in one order only, and none in both orders with different graphs. So the clauses of the rule about what one edit creates and the other needs (`C ∩ L`) never mattered in these runs.

## The exhaustive pass (added after the result, not pre-registered)
Because every line came out as predicted I asked a stricter question of the rule: is it exact on EVERY pair, in a universe small enough to list them all? `python/axi/experiments/swarm_exhaustive.py` (test 110): three cells; the 80 graphs on them (every subset of the cells as services, every set of directed `dep` edges between them); on each graph, every edit of one to three operations that the gate accepts (add a service with value 1 or 2, delete one, add an edge with weight 1 or 2, delete one, in every combination); every ordered pair of two of those edits, including an edit with itself. **1,262,688 ordered pairs, of which 1,075,824 do not commute** (everything overlaps in a universe this small).

| detector | caught | missed | false alarms |
|---|---|---|---|
| `dpo` (the rule) | 1,075,824 | **0** | **0** |
| `pager` | 1,024,427 | 51,397 (4.8 %) | 137,912 (73.8 % of the 186,864 pairs that commute) |
| `names` | 1,013,848 | 61,976 (5.8 %) | 0 |

The rule is exact on this universe: not one pair misjudged, in either direction. The rates of the other two are not comparable with the random runs (the universe is crowded); what carries over is the kind of error (`pager`: edge additions that share a service; edge deletions that share an edge or its service). The result file is `docs/results/axi_swarm_exhaustive.json` (50 seconds). **This is exactness on three services and edits of at most three operations, not a proof for the language**: a service with three edges cannot be deleted by an edit of three operations, so the compound edits of the random runs are out of reach here; and the edit language has no in-place update.

## What this does and does not show about the text's claims
* **"10,000 AI agents can read and write to the exact same enterprise state graph simultaneously using Copy-on-Write (CoW) shadow paging."** Not tested at 10,000: the largest round has 64 agents and the threaded runs have 8. At 64 agents on 40 services only about 70 % of a simultaneous batch can be taken together, at 640 services 96 %. As a statement about copy-on-write shadow paging it does not hold for the pager as written (item 3): the paging does not decide what may be taken together.
* **"Without AXI, multi-agent swarms inevitably overwrite each other's context and corrupt the shared environment."** In this model, with no gate, 52 to 68 % of rounds of 16 agents ended with an edge to a service that no longer exists, and a store in which the last writer keeps the whole state lost updates (line 4). "Without AXI" means without the gate here; "inevitably" and "context" are not measured.
* **"AXI acts as the atomic transaction manager for the swarm."** For the gate on the current state at commit time, and for the independence rule, this is what the runs show (lines 5, 6, 10, 16, 17 and the exhaustive pass), under eight Python threads. Not shown: atomicity under hardware races or on several machines, the Rust crate, and the "O(1) pointer swap".
* **The statelessness that the claim starts from** was not tested; it is a claim about what a model needs to read, not about what the store does.

**Not done and worth doing:** a second, independently written implementation of the truth and of the rule (mine are written by the same hand); an edit kind that updates in place (the rule would need a read-write clause); more services in the exhaustive pass; processes instead of threads.

## Reproduce
In `python/`: `python -m axi.experiments.swarm --out axi_swarm.json` (about two minutes on a laptop). The printed digest should be **dd08fd515ff59bff** whatever the machine (it matched under Python 3.11 and 3.13; the quick run, `--quick`, matched under 3.11, 3.12 and 3.13); the counts of the threaded runs will differ slightly, the 19 lines should not. The exhaustive pass: `python -m axi.experiments.swarm_exhaustive --out axi_swarm_exhaustive.json` (about a minute).
