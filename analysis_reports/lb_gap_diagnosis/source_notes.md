# Source notes

## Report job

- Question: Why did exp05 score about 0.77 on the Public LB despite nested OOF F1 0.8063?
- Audience: product stakeholders / competition team.
- Decision: determine whether to trust exp05 and how to change the evaluation workflow.
- Comparison basis: exp08 k=0–15 fixed-split nested OOF results versus the user-reported Public LB ≈0.77.

## Required-structure mapping

- Title: `CVとLBの乖離診断`
- Executive Summary: visible immediately after the title.
- Key findings: `0.8063は独立な検証値ではない`, chart, and driver assessment table.
- Recommended next steps: `次の実験では評価系を先に固定する`.
- Further questions: Public subset size, historical LB results, Public positive prevalence.
- Caveats and assumptions: visible final section.

## Report spine

- Answer: adaptive reuse of the same outer CV for feature-count/model selection is the primary driver; finite-sample variance is the second driver.
- Validation: threshold sensitivity, selected-feature train/test SMD and KS, OOF bootstrap interval.
- Rejected as primary explanations: threshold 0.247 and large univariate shift in the two selected tree features.
- Next step: inner-loop model selection or untouched holdout, repeated seeds, and fold-model ensembling.

## Chart map

- Segment: selection bias.
- Question: was 0.8063 a stable plateau or the maximum of many tried candidates?
- Type: two-series line chart.
- Fields: k, series, f1.
- Claim: k=2 is a local peak selected from 16 candidates; the unseen Public LB did not reproduce it.
- QA: exact values are retained in the bounded snapshot; the LB line is explicitly labeled approximate and user-reported.

## Evidence and limitations

- `analyze.py` recomputes diagnostics from repository data and code.
- Public LB ≈0.77 is user-reported; leaderboard subset size and labels are unavailable.
- Bootstrap is conditional on one OOF prediction vector and does not incorporate adaptive-selection uncertainty.
- Univariate drift checks cover the two added features, not the full joint feature distribution.
