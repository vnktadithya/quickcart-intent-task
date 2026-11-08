# DECISIONS — QuickCart intent task

One entry per decision. No entry without evidence.

## Template
### D<n> — <short title>
- **Phase:**
- **Question:**
- **Options considered:**
- **Evidence:** (numbers + file)
- **Decision:**
- **What would change it:**

## D0 — Choose models on a checked dev subset, not silver dev
- **Phase:** planning
- **Question:** what set decides the final model?
- **Options considered:** silver dev (4,000 rows); hand-checked dev subset (~250 rows).
- **Evidence:** silver labels contain non-random errors (README; examples in NOTES.md #11). A model trained on silver train scores 0.86 on silver dev partly by reproducing the same errors.
- **Decision:** select on the checked subset; report silver-dev numbers beside it.
- **What would change it:** the checked subset agreeing with silver dev above ~95%, which would make silver dev a fair judge.
