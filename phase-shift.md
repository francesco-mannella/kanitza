# Gabor phase offset: comparison of the original and the corrected front end

Record of the comparison only. The draft, the artifact and the plan describe the corrected front end (`gabor_phase_offset` = -pi/2) as the champion configuration and make no reference to the original.

## Finding

The original champion used `gabor_phase_offset` = -1.4828 rad, not exactly -pi/2. The kernels are nearly odd and the response is rectified (`abs` of the convolution), so the small even component makes edge saliency depend on edge polarity. Red square on black, edge midpoints (corners excluded):

| Phase offset | top | bottom | left | right |
|---|---|---|---|---|
| -1.4828 | 6.61 | 3.97 | 5.50 | 4.41 |
| -pi/2 | 6.44 | 6.44 | 6.44 | 6.44 |

The executed saccade is the most salient pixel under the Gaussian attention mask (sampling precision 0.999999), so it follows a command only where something salient is. With the original offset the orienting saccade also lands on the top-left of the object, so the object lies mostly below and to the right of the fixation. Downward commands were often executed as something else, while the attention map was trained on the command: the training pair (command, executed saccade, content shift) was inconsistent in y. Not a code bug (agent replica reproduces the executed y, corr 0.99; the mask is symmetric; the y flip is applied once).

Counterfactual on 1200 replayed saccades (seed 16180, original offset), executed y from the saliency averaged with its vertical mirror: corr with the command 0.58 -> 0.86, executed in the commanded direction 62% -> 91%, mean executed y -0.3 -> -14.6 (the mean command is -13). Mirroring in x only changes x slightly (0.89 -> 0.89).

## Logged displacements (`log_displacement`, window 48, last 100 epochs, salient samples)

| Seed | corr(commanded, executed) y, original | y, corrected | executed in commanded direction y, original | corrected |
|---|---|---|---|---|
| 031415 | -0.75 | -0.98 | 69% | 98% |
| 016180 | -0.74 | -0.91 | 61% | 92% |
| 054321 | -0.78 | -0.94 | 66% | 99% |
| 027182 | not logged | -0.94 | | |
| 012345 | not logged | -0.98 | | |

(Sign negative by the retina convention.) x correlation 0.93 in the original runs, 0.92 to 0.98 corrected. Share of downward commands 83-86% original, 50-51% corrected. Executed amplitude per axis, original: x about 20, y about 10 retina px; corrected: balanced, split varies by seed (x 10.7 to 21.3, y 12.7 to 22.7). The retrain of seed 16180 with logging reproduces the original run bit for bit (all weights differ by 0.0): training is deterministic per seed, so the logged displacements describe the original runs.

## Window 48, normal pairing, five confirmatory seeds

| Seed | Landing r, original | corrected | Competence (last 50), original | corrected | -b/a (peak), corrected |
|---|---|---|---|---|---|
| 031415 | 0.16 | 0.56 | 0.758 | 0.867 | 0.51 |
| 016180 | 0.09 | 0.82 | 0.759 | 0.834 | 1.35 |
| 054321 | -0.05 | 0.76 | 0.751 | 0.889 | 0.91 |
| 027182 | 0.46 | 0.48 | 0.806 | 0.865 | 0.30 |
| 012345 | 0.56 | 0.58 | 0.780 | 0.897 | 0.70 |

Corrected: permutation null about 0, p <= 0.01 in all five; rule 2.6(a) holds in 5 of 5 (original: 2 of 5). Mean competence 0.771 -> 0.870. Median -b/a 0.70 corrected (0.02 original). Cosine between field shift (cc - ce) and d: 0.89-0.98, p = 0.002 in all five.

Scanpaths (corrected, 5 runs): cosine executed vs commanded 0.83-0.98 (original 0.75-0.95); the effects prototype predicts the landing view better than the conditions prototype in 91-99% of saccades (rank 16-27 against 58-66); revisit share 0.82-0.86 (original 0.84-0.87). C1/C2 of the paradigm still FAIL (median FF-type 0.51, control 0.48), as before.

## Retrains with the corrected offset

| Folder (under tests) | Content | Status |
|---|---|---|
| `phase_offset_2026-10-10` | 5 confirmatory seeds, window 48, normal pairing, `log_displacement` | done, analysed |
| `phase_fixed_confirm_2026-10-10` | same seeds, window 48, shuffled pairing | training |
| `phase_fixed_pilot_2026-10-10` | pilot seeds 14142, 17320, 22360: window 48 and 64, normal and shuffled | training |
| `phase_fixed_champion_2026-10-10` | window 16, seeds 039973, 090902, 093581, 027182, 031415 | training |

Other post hoc runs under the original offset, not repeated: `inhibition_radius_2026-10-09` (goal inhibition std 1 and 2, finished, not analysed), `displacement_log_2026-10-10` (logging reruns of 031415, 016180, 054321).

Scripts: `exp_stimulus_landing.py`, `exp_pairing_regression.py`, `exp_remapping_paradigm.py`, `exp_scanpaths.py`, `write_goals.py`; `src/test.py --plot` for the demo gifs.

## Window 16 under the corrected offset

Five window-16 runs (`phase_fixed_champion_2026-10-10`, seeds 039973, 090902, 093581, 027182, 031415) retain the retinotopy (spot probe z > 3 in 10 of 10 maps, median rho 0.92 conditions and 0.89 effects, median z 38.8 and 36.0; phase-encoded z > 3 in 10 of 10, rho 0.48 and 0.53, shuffled-layout control not exceeded, p 0.63 to 1.0) but train poorly: competence (last 50 epochs) 0.29 to 0.32, executed saccades 8.2 to 9.1 retina px with commands of about 7 px, cosine executed vs commanded 0.25 to 0.41, effects prototype better than conditions prototype for the landing view in 41 to 50% of saccades. The original window-16 champion (same five seeds) had competence 0.64 to 0.75 and executed amplitudes 11 to 29 px (cosine 0.32 to 0.90). Cause not investigated. Decision: the standard window becomes 48 or 64; Phase 1 is repeated on the window-48 and window-64 runs (`tests/phase1_standard_2026-10-10`).

## Shuffled pairing under the corrected offset (window 48, five confirmatory seeds)

Landing r normal / shuffled: 031415 0.56 / -0.44; 016180 0.82 / -0.01; 027182 0.48 / -0.26; 012345 0.58 / 0.48; 054321 0.76 / 0.73. Competence shuffled 0.57 to 0.67. -b/a shuffled 0.04, -0.05, 0.15, 0.05, 0.23 (normal median 0.70); cosine (cc - ce, d) shuffled 0.39 to 0.94. Rule 2.6(b) holds (r(shuf) < r(unshuf) in every seed, median r(shuf) -0.01 against a normal median of 0.58), but in two seeds (012345, 054321) the drop is small (0.10 and 0.03). Pilot seeds, normal training: window 48 r 0.30, 0.45, 0.41 (p 0.07, 0.02, 0.03), window 64 r 0.72, 0.84, 0.55 (all p < 0.01).

Pilot seeds, shuffled pairing: window 48 r 0.16 (p 0.08), 0.02, -0.35 (normal 0.30, 0.45, 0.41); window 64 r -0.71, -0.05, 0.29 (p < 0.01) (normal 0.72, 0.84, 0.55). Over the 11 normal/shuffled pairs shuffling lowers r in all and leaves a large part in three (0.48, 0.73, 0.29). Median -b/a (peak): normal 0.49 at window 48 (eight seeds), 0.28 at window 64; shuffled 0.05 (five confirmatory seeds).

Figures of the draft (maps, scanpaths, landing example) use the window-64 run pf_unshuf64_014142 (best worst case of conditions ordering 0.85, effects ordering 0.82, agreement 0.91 among the three window-64 runs); landing_pairing.pdf is made by `scripts/fig_landing_pairing.py` from the landing CSVs of the eleven normal/shuffled pairs.

## Phase 1 and scanpath measures on the standard windows (eleven normal runs, corrected offset)

`indirect_retinotopy_test.py` (spot probe, soft readout): z > 3 in 22 of 22 maps, median rho 0.73 (conditions) and 0.74 (effects), median z 37.2 and 36.7, shuffled-layout control exceeded in 22 of 22 (p <= 0.03), range <= 0.5 in 22 of 22. Phase-encoded probe: z > 3 in 22 of 22, median rho 0.57 and 0.52, p < 0.05 in 21 of 22. Scanpaths (goals files, `exp_scanpaths.py`, purity as in `results_server.py`; test without `exclude_fixation`, see the section below): kNN purity 0.83 to 0.98 (mean 0.89), 6.9 distinct goals per test (5.7 to 9.2), distinct-shape pairs sharing a goal 0.29 (0.18 to 0.43), 14 to 18 distinct first goals, competence 0.76 to 0.90 (mean 0.86), cosine executed vs commanded 0.92 to 1.00, revisit share 0.76 to 0.85. Paradigm C1 (share of future-field responses against the permuted-effects control): median FF-type 0.56 against 0.50 at window 48 (eight runs), 0.74 against 0.69 at 64 (three), margin smaller than the seed SD (0.07 to 0.13): FAIL, as before; C2 FAIL. Figures of the draft regenerated from these runs (`docs/figures`: maps_champion, scanpath_champion, landing_example, landing_pairing, retinotopy_probe_{maps,null,stats}).

## Test-time `exclude_fixation` at the standard windows

Scanpath tests had been run with `exclude_fixation` forced on (the old champion test protocol, written into `write_goals.py` and `exp_scanpaths.py`). The exclusion radius is `fovea_scale`/2 retina px: 8 at the 16 px window, 24 and 32 at 48 and 64 px on an 80 px retina, so most of the retina is blanked and the saccade goes to what is left. In the draft figure (window 64, seed 014142) fixations then fell up to about 15 px outside the object. Comparison on 8 runs (5 at 48, 3 at 64), 18 tests each (`tests/scanpath_B_2026-10-10`, scratch script `ab.py`):

| Test setting | Purity | Goals/test | Sharing | Undone goal saccades | Steps >2 px off the object |
|---|---|---|---|---|---|
| radius `fovea_scale`/2 (as run before) | 0.83 | 6.1 | 0.24 | 0 | 2% (6% in two 64 px runs) |
| A: radius 8 px | 0.88 | 6.9 | 0.28 | 0 | 0% |
| B: exclusion off (run's own setting) | 0.89 | 6.9 | 0.28 | 0 | 0% |

Decision: B. `hold_fixation` already prevents undone goal saccades, and B equals A. `exp_scanpaths.py` now takes the run's own setting (`--exclude-fixation` forces it on); `write_goals.py` is called with `--no-exclude-fixation`. Earlier goals files are kept in `goals_exclfix/` of each run folder. Fixations lie on object edges (saliency is the absolute Gabor response, so the maximum is within 1 px of the contour, on either side). Figure `scanpath_champion.pdf` and the scanpath numbers in the draft were regenerated with B. The landing, pairing, retinotopy and paradigm statistics do not use test scanpaths and are unchanged.

## Scanpath measures of the shuffled-pairing runs (exclusion off, 11 runs)

Same tests as the normal runs (`tests/scanpath_B_2026-10-10`, `manifest_shuf.csv`, `scanpaths_summary_manifest_shuf.csv`; goals files regenerated with `write_goals.py --no-exclude-fixation`, old ones in `goals_exclfix/`). Min / mean / max over the 11 runs, normal against shuffled:

| Measure | Normal | Shuffled |
|---|---|---|
| kNN purity | 0.83 / 0.89 / 0.98 | 0.53 / 0.74 / 0.91 |
| Distinct goals per test | 5.7 / 6.9 / 9.2 | 5.9 / 7.0 / 8.4 |
| Pairs sharing a goal | 0.18 / 0.29 / 0.43 | 0.31 / 0.39 / 0.47 |
| Competence (last 50 epochs) | 0.76 / 0.86 / 0.90 | 0.38 / 0.63 / 0.67 |
| Cosine executed vs commanded | 0.92 / 0.97 / 1.00 | 0.73 / 0.96 / 1.00 |
| Effects prototype better than conditions prototype | 0.65 / 0.90 / 1.00 | 0.50 / 0.68 / 0.88 |
| Rank of the effects prototype for the landing view (of 100) | 5 / 20 / 33 | 17 / 49 / 65 |
| Rank of the conditions prototype | 50 / 63 / 80 | 43 / 58 / 69 |
| Revisit share | 0.76 / 0.82 / 0.85 | 0.71 / 0.82 / 0.85 |
| Executed amplitude (retina px) | 21.9 / 26.8 / 29.6 | 11.1 / 26.0 / 29.8 |

Per run, effects better than conditions / rank of the effects prototype, shuffled: 012345 0.76 / 46.8; 016180 0.79 / 46.5; 027182 0.64 / 48.1; 031415 0.50 / 64.9; 054321 0.88 / 16.6 (landing r 0.73 under shuffling); pilot 48: 0.65 / 47.0, 0.66 / 48.2, 0.71 / 49.9; pilot 64: 0.79 / 61.5, 0.58 / 60.5, 0.54 / 47.4. The 64 px seed 022360 shuffled run executes short saccades (11.1 px against 12.1 commanded, cosine 0.73; competence 0.38). Shuffling lowers purity and competence and removes most of the effects prediction (except 054321), and leaves the direction and size of the executed saccades unchanged in 10 of 11 runs.

