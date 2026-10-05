# Within-dataset control — pre-registration

Written before the control was run, so the reading cannot be chosen after
seeing the numbers.

Recorded: 2026-09-18.

## The gap this closes

Adversarial validation separates CICIDS2017 from CSE-CIC-IDS2018 at AUC
0.9994–1.0000 inside every testable class. That was called a *dataset
fingerprint*. The existing null control — one corpus split into random halves —
returns 0.46–0.53, which proves only that the **procedure** cannot invent
separation out of nothing. It proves nothing about the **claim**, because a
random split destroys exactly the structure at issue: shuffling rows mixes
every capture day and every hour into both halves, so a difference that tracks
*when* or *in which file* a flow was captured cannot survive it.

So the two explanations still standing are:

1. **Dataset fingerprint.** Something about how each corpus was produced — the
   lab, the tooling, the generation scripts — marks its rows, and that mark
   does not exist between two captures of the same corpus.
2. **File/time effect.** Traffic simply differs between capture sessions.
   Monday differs from Friday inside 2017 as much as 2017 differs from 2018,
   and the cross-dataset AUC is measuring the ordinary passage of time rather
   than anything about the datasets.

These predict opposite results for a split that *keeps* the file and time
structure instead of shuffling it away.

## Design

For each shared class, the identical procedure — balanced draw, LightGBM on
raw columns, 70/30 stratified split, five seeds, the same 60 collapsed
features — is run on six contrasts:

| contrast | side 0 | side 1 |
| --- | --- | --- |
| `cross` | that class in 2017 | that class in 2018 |
| `time_ids2017` | earliest half by `_row_index` | latest half |
| `time_ids2018` | earliest half by `Timestamp` | latest half |
| `group_ids2017` | one set of capture files | the rest |
| `group_ids2018` | one set of capture days | the rest |
| `null_ids2017` / `null_ids2018` | a random half | the other half |

**Matched n.** Every contrast for a class uses the *same* number of rows per
side — the smallest any available contrast can supply. AUC depends on sample
size, so comparing a cross-dataset draw of 8,000 a side against a within-dataset
draw of 400 would confound the thing being measured with the thing being
controlled for.

**Floor of 300 rows a side**, lower than the 500 the main run uses. The risk at
small n is that a model *memorises* and the AUC comes out too high; that works
against the fingerprint conclusion rather than for it, so a small-n control is
conservative here in a way it would not be in the main run.

**Groups are assigned to sides by greedy size balancing**, not chronologically.
Splitting files in time order would rebuild the time contrast under another
name; balancing puts early and late captures on both sides, so what is left to
separate is file identity rather than date.

## Known confound, stated in advance

In both corpora a capture file *is* a capture day. A group split is therefore
also a time split at coarser granularity. The two within-dataset contrasts
differ in granularity, not in kind, and neither isolates "file" from "day".
What they jointly establish is whether *any* within-corpus capture boundary
separates as strongly as the corpus boundary does — which is the question that
matters here.

## Decision rule

Consider the classes where the cross-dataset AUC is ≥ 0.99. Let the
within-dataset AUC for a class be the **highest** of its four within-dataset
contrasts, so the comparison is against the strongest case for the rival
explanation rather than the most convenient one.

* **Fingerprint confirmed** — in every one of those classes, the within-dataset
  AUC is at least 0.10 below the cross-dataset AUC *and* below 0.95.
* **Fingerprint refuted** — the within-dataset AUC reaches ≥ 0.99 in a majority
  of them. What was attributed to the datasets is a capture-session effect.
* **Partial** — anything else. Report the per-class table and claim nothing
  beyond it.

The thresholds are anchored on numbers already on record rather than chosen
round: 0.50 is where the random null sits, 0.766 is the best any single column
manages within Benign, and 0.99 is where every cross-dataset class already is.

## Prediction

Partial, leaning confirmed. Benign is the class most likely to separate
within-corpus — it spans every capture day in both corpora, and weekday traffic
genuinely differs — so a within-dataset AUC well above chance is expected there
and would not refute anything on its own. The attack classes are the test: in
2017 each lives in exactly one capture file, so a group split is impossible for
them and only the time contrast applies, splitting a single attack window in
half. If the corpus boundary is doing real work, splitting one attack window
should separate far more weakly than splitting across corpora does.
