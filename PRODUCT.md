# Product

## Register

product

## Users

Four audiences, in descending order of how much the design has to carry them:

- **Examining committee / faculty who have not read the code.** They meet this
  interface once, for a few minutes, and have to leave understanding what was
  measured and how far it can be trusted. They will not open a terminal, read a
  notebook, or ask a clarifying question before forming an opinion.
- **The supervising professor.** Returns repeatedly, already knows the domain,
  and probes: why this model, why that split, why is that class empty. The
  interface has to survive follow-up questions rather than just look complete.
- **The project team (2 people).** Daily working tool while iterating on models
  and sample sizes. Needs speed and precision over explanation.
- **A SOC analyst, hypothetically.** Not a real user today, but the framing that
  keeps the metrics honest: the numbers that matter operationally are
  false-alarm rate and missed attacks, not headline accuracy.

## Product Purpose

A dashboard over network-intrusion-detection results produced by two separate
pipelines — CICIDS2017 and CSE-CIC-IDS2018 — that were trained under different
protocols, class taxonomies, and evaluation regimes.

**The primary job is comparison across those two datasets.** Everything else
(per-model detail, per-class breakdowns, batch prediction) exists to support
that comparison or to answer the questions it provokes.

Success is a viewer who leaves with a correct belief rather than an impressed
one. Concretely: they can state which model won, by how much, whether that
margin is real, and which numbers on screen are not measurements at all.

The comparison is genuinely difficult and the interface must not paper over it.
The two bundles disagree about the split protocol (temporal vs random), the
class count (9 vs 15), whether hyperparameters were tuned, and which metrics
were recorded at all. A UI that renders them as two interchangeable tabs of the
same shape would be lying.

## Brand Personality

**Research-grade, legible, auditable.**

The voice is a good methods section: it states what was done, what was found,
and what the finding does not support — in that order, without hedging and
without overclaiming. It never sells. When a result is weak it says so in the
same typeface and the same tone as when a result is strong.

Emotionally the target is *earned confidence*. A viewer should trust the numbers
because the interface keeps showing its work, not because it looks polished.

## Anti-references

- **Generic student project.** Default Bootstrap, default Streamlit, unstyled
  tables, a scatter of charts with no argument connecting them. This is the
  explicit thing to avoid; the work behind this project is not undergraduate-
  average and the interface should not imply that it is.
- **Template admin dashboards.** Decorative donuts, gauges, and sparklines
  placed because a dashboard is supposed to have them. Every chart must answer a
  question someone actually asked.
- **SaaS hero-metric pages.** One enormous number over a gradient, small label
  beneath, supporting stats in a row. Looks decisive, communicates nothing about
  reliability.
- **Unqualified claims.** F1 0.99 set in large type with no indication that the
  class has one test flow behind it. This is the specific failure mode this
  domain invites, and it is the one most likely to mislead a committee.

## Design Principles

1. **Absent is not zero.** A metric the bundle never recorded renders as absent,
   never as 0. A false-positive rate of 0.00000 is the best score a detector can
   post; a false-positive rate that was never measured tells you nothing.
   Collapsing the second into the first makes the least-evaluated model look
   like the best one.

2. **Support travels with the score.** No per-class metric appears without the
   number of test flows behind it. Three classes in the 2018 bundle have 1–3
   test flows and two in the 2017 bundle have 4 and 11; scores computed on those
   describe the sample, not the model, and the interface must make that
   impossible to miss.

3. **Provenance is part of the number.** Split protocol, class weighting and
   tuning state change what a score means, so they stay visible next to the
   scores rather than living on a separate "about this run" page. Switching
   bundles must visibly change the frame, not just the digits.

4. **Comparison is the product.** The primary job is 2017 versus 2018. Any
   design that makes the two harder to hold in mind at once — burying the
   switch, resetting state on change, showing them in incomparable units — has
   failed at the main thing regardless of how it looks.

5. **Legible cold, in one pass.** A committee member with no context must reach
   a correct reading without asking a question. This constrains vocabulary,
   labelling and information order; it does not license dumbing the numbers
   down.

## Accessibility & Inclusion

- **WCAG 2.1 AA.** Body text ≥ 4.5:1, large text ≥ 3:1, visible focus states,
  full keyboard operation of the bundle switcher, model selector and tables.
- **Colour is never the only channel.** The dashboard already leans on a
  semantic colour ramp (ok / info / warn / danger) for status, and on a
  cyan-intensity confusion matrix. Every such signal needs a redundant
  encoding — text, numeral, or icon — so it survives colour-vision deficiency
  and grayscale printing. Confusion-matrix cells therefore carry their
  percentage as text, not just as saturation.
- **Reduced motion respected.** Any transition has a
  `prefers-reduced-motion: reduce` path.
- **Presentation conditions.** This will be shown on a projector in a lit room.
  Contrast has to hold up well past the AA floor at small sizes; the dark theme
  cannot rely on subtle low-contrast separators to carry structure.
