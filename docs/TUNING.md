# Tuning on dev units

Rendered from `benchmarks/tuned.json` by `scripts/tune.py`; the protocol is `docs/adr/0003-tuning-protocol.md`, written before this run. C-MAPSS is simulated by NASA's C-MAPSS model, not field data.

- **Data:** 195 dev units of FD001 to FD004 only. Dev list SHA-256 `b14e28acb7b5202740ecd7faffe971d54c39afbbac2db40181fca49ae7e8f04a`. No calib, eval or NASA test data was read.
- **Seeds:** key `tune`, apart from the full run's.
- **Chosen:** window 50, cap 125, s_min 2; h 5; M 35.
- **Note:** h limit met.

## Estimator grid (V1 on out-of-fold predictions; one drawn row per unit)

Width and coverage are means over every row of every dev unit. Coverage is not part of the criterion.

| window | cap | s_min | mean width | pooled coverage | RMSE of m | q_lo | q_hi |
|---|---|---|---|---|---|---|---|
| 15 | 125 | 2 | 48.4825 | 0.9140 | 16.7781 | -2.1021 | 2.0649 |
| 15 | 125 | 5 | 45.4837 | 0.9007 | 16.7781 | -1.9841 | 1.6907 |
| 30 | 125 | 2 | 41.1295 | 0.9122 | 14.8733 | -2.2948 | 1.9408 |
| 30 | 125 | 5 | 40.9002 | 0.9100 | 14.8733 | -2.0599 | 1.8403 |
| 50 | 125 | 2 | 38.2394 | 0.9251 | 13.5529 | -2.2791 | 2.1602 |
| 50 | 125 | 5 | 39.8396 | 0.9343 | 13.5529 | -2.0973 | 2.0814 |

## Threshold h (closed loop, V1, chosen estimator)

Limit: unplanned-failure fraction at most 0.05. Limit met: yes.

| h | failed fraction | replacements | mean RUL at replacement |
|---|---|---|---|
| 0 | 0.0615 | 183 | 13.7104 |
| 5 | 0.0000 | 195 | 19.1744 |
| 10 | 0.0000 | 195 | 26.7949 |
| 15 | 0.0000 | 195 | 35.3641 |
| 20 | 0.0000 | 195 | 43.4103 |
| 25 | 0.0000 | 195 | 52.1590 |
| 30 | 0.0000 | 195 | 60.3641 |

## Stratum M (closed loop at h = 5)

The smallest value with at least 90% of dev units having a row with m <= M before replacement. Values are listed up to the one chosen.

| M | units with stratum rows | fraction |
|---|---|---|
| 5 | 0 | 0.0000 |
| 10 | 73 | 0.3744 |
| 15 | 152 | 0.7795 |
| 20 | 165 | 0.8462 |
| 25 | 170 | 0.8718 |
| 30 | 171 | 0.8769 |
| 35 | 176 | 0.9026 |
