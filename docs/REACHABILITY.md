# Reachability of the primary verdicts

Made-up inputs built from dev units only (C-MAPSS, simulated by NASA's C-MAPSS model). The per-unit values of each operating point are resampled to the registered size and run through the full run's interval and verdict code at level 0.008333 (0.05 / 6). These check the rules and the sizes; they are not results.

## Operating point: quick

- Input `benchmarks/.cache/quick.json`, SHA-256 `45dd81e3c3d39d45725278d4b3306f980312502356bb9fa66819b511ca837619`; made with code hash `79d4b7a8ebe6e205377535ed946b8b9f7ab9e56020182880ed11307575fb10a0`.
- 35 dev units' per-unit values, resampled to the registered size 20 times. Labels are counts over the resamples.

### C2-PERC (band +-0.03)

Uniform shift (every unit moves by the same amount):

| row | dev units | dev mean | dev sd | kurtosis | n | guard (n/50) | median half-width | method | as is | gap 0 | gap -0.03 | smallest winning gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-PERC V1, pooled | 35 | -0.0030 | 0.0120 | 19.15 | 373 | 7.46 (fails) | 0.0197 | betting | loss 20 | loss 20 | inconclusive 20 | -0.055 |
| C2-PERC V1, stratum | 7 | 0.0000 | 0.0000 | n/a | 74 | 1.48 (fails) | 0.0943 | betting | inconclusive 20 | inconclusive 20 | inconclusive 20 | -0.140 |
| C2-PERC V2L, pooled | 35 | -0.0023 | 0.0070 | 16.69 | 373 | 7.46 (fails) | 0.0197 | betting | loss 20 | loss 20 | inconclusive 20 | -0.055 |
| C2-PERC V2L, stratum | 11 | 0.0000 | 0.0000 | n/a | 117 | 2.34 (fails) | 0.0610 | betting | inconclusive 20 | inconclusive 20 | inconclusive 20 | -0.100 |

Concentrated shift (the whole gap carried by 10% of the units, floored at 0; 'reached' is the median mean difference the shift reached):

| row | gap 0 | gap -0.03 | smallest winning gap (reached) | at -0.055 | at -0.08 | at -0.1 |
|---|---|---|---|---|---|---|
| C2-PERC V1, pooled | loss 20 | inconclusive 20 | -0.060 (-0.0600) | inconclusive 20 (reached -0.0550, half-width 0.0293) | win 20 (reached -0.0800, half-width 0.0417) | win 20 (reached -0.0997, half-width 0.0519) |
| C2-PERC V1, stratum | inconclusive 20 | inconclusive 20 | n/a (n/a) | inconclusive 20 (reached -0.0550, half-width 0.1040) | inconclusive 20 (reached -0.0800, half-width 0.1175) | inconclusive 20 (reached -0.1000, half-width 0.1371) |
| C2-PERC V2L, pooled | loss 20 | inconclusive 20 | -0.060 (-0.0600) | inconclusive 20 (reached -0.0550, half-width 0.0295) | win 20 (reached -0.0800, half-width 0.0420) | win 20 (reached -0.0993, half-width 0.0522) |
| C2-PERC V2L, stratum | inconclusive 20 | inconclusive 20 | n/a (n/a) | inconclusive 20 (reached -0.0550, half-width 0.0706) | inconclusive 20 (reached -0.0800, half-width 0.0825) | inconclusive 20 (reached -0.1000, half-width 0.0993) |

### C2-SELECT (margin 0.02, floor 0.92)

- Dev units 35; difference mean 0.0034, sd 0.0127, kurtosis 23.93; guard at n = 373: fails.
- VW closed-loop c^lo, dev mean 0.9952; median lower end at n = 373: 0.9760. Median half-width of the difference: 0.0197 (betting).
- Labels as they are: inconclusive 20.
- Uniform shift: at a difference of 0, loss 20; smallest difference with a win in every resample: 0.045.
- Concentrated shift (10% of units): at a difference of 0, loss 20; smallest difference with a win in every resample: 0.045 (reached 0.0450).
- VW's fixed-point steps in this input ended: two-cycle. These rows use VW's last pair; a C2-SELECT win or loss also needs VW settled, or both pairs of a two-cycle giving the same label.

## Operating point: dev-point

- Input `benchmarks/.cache/dev-point.json`, SHA-256 `39bab016e7a362987531c821b284a9476180f3aeba627b8fa8ec3f3607ff4bac`; made with code hash `bb0cc7a149c9c6345decde1d63a2aeaf562c2db9d532a1368ea4d04dc32ee26c`.
- 195 dev units' per-unit values, resampled to the registered size 20 times. Labels are counts over the resamples.

### C2-PERC (band +-0.03)

Uniform shift (every unit moves by the same amount):

| row | dev units | dev mean | dev sd | kurtosis | n | guard (n/50) | median half-width | method | as is | gap 0 | gap -0.03 | smallest winning gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2-PERC V1, pooled | 195 | -0.0057 | 0.0247 | 35.21 | 373 | 7.46 (fails) | 0.0199 | betting | loss 20 | loss 20 | inconclusive 20 | -0.055 |
| C2-PERC V1, stratum | 127 | 0.0017 | 0.0175 | 56.37 | 242 | 4.84 (fails) | 0.0302 | betting | inconclusive 20 | inconclusive 20 | inconclusive 20 | -0.065 |
| C2-PERC V2L, pooled | 195 | -0.0342 | 0.0429 | 6.16 | 373 | 7.46 (holds) | 0.0058 | t | inconclusive 17, win 3 | loss 20 | inconclusive 20 | -0.040 |
| C2-PERC V2L, stratum | 140 | -0.0422 | 0.0769 | 6.44 | 267 | 5.34 (fails) | 0.0293 | betting/t | inconclusive 18, win 2 | inconclusive 10, loss 10 | inconclusive 20 | -0.065 |

Concentrated shift (the whole gap carried by 10% of the units, floored at 0; 'reached' is the median mean difference the shift reached):

| row | gap 0 | gap -0.03 | smallest winning gap (reached) | at -0.055 | at -0.08 | at -0.1 |
|---|---|---|---|---|---|---|
| C2-PERC V1, pooled | loss 20 | inconclusive 20 | -0.060 (-0.0600) | inconclusive 18, win 2 (reached -0.0550, half-width 0.0282) | win 20 (reached -0.0793, half-width 0.0400) | win 20 (reached -0.0981, half-width 0.0495) |
| C2-PERC V1, stratum | inconclusive 20 | inconclusive 20 | -0.080 (-0.0800) | inconclusive 20 (reached -0.0550, half-width 0.0413) | win 20 (reached -0.0800, half-width 0.0569) | win 20 (reached -0.0999, half-width 0.0711) |
| C2-PERC V2L, pooled | loss 20 | inconclusive 20 | -0.040 (-0.0400) | win 20 (reached -0.0550, half-width 0.0104) | win 20 (reached -0.0800, half-width 0.0196) | win 20 (reached -0.0998, half-width 0.0367) |
| C2-PERC V2L, stratum | inconclusive 20 | inconclusive 20 | -0.065 (-0.0650) | inconclusive 8, win 12 (reached -0.0550, half-width 0.0144) | win 20 (reached -0.0800, half-width 0.0282) | win 20 (reached -0.1000, half-width 0.0402) |

### C2-SELECT (margin 0.02, floor 0.92)

- Dev units 195; difference mean -0.0052, sd 0.0277, kurtosis 19.29; guard at n = 373: fails.
- VW closed-loop c^lo, dev mean 0.9656; median lower end at n = 373: 0.9429. Median half-width of the difference: 0.0200 (betting).
- Labels as they are: loss 20.
- Uniform shift: at a difference of 0, inconclusive 20; smallest difference with a win in every resample: 0.045.
- Concentrated shift (10% of units): at a difference of 0, inconclusive 20; smallest difference with a win in every resample: 0.050 (reached 0.0500).
- VW's fixed-point steps in this input ended: two-cycle, settled. These rows use VW's last pair; a C2-SELECT win or loss also needs VW settled, or both pairs of a two-cycle giving the same label.

## C2-EXT ([0.92, 0.98], Clopper-Pearson at 707 units; no dev values needed)

| coverage | k | low | high | label |
|---|---|---|---|---|
| 0.95 | 672 | 0.9249 | 0.9695 | win |
| 0.92 | 650 | 0.8886 | 0.9441 | inconclusive |
| 0.85 | 601 | 0.8115 | 0.8836 | loss |
