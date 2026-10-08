# Champion configuration

Reconstructed from the sweep history (`docs/report.tex`, memory notes, `tests/hparam_*/*_summary.csv`). The report stops before the training-time `orient` and `exclfix` sweeps, so the choice below rests on the summary CSVs and follows the user's scanpath criteria (object regions separated, no goal overlap between rotations) plus faithful execution of every goal saccade.

## Choice

**Training**: `orientT_ior`: reference parameters (anchor_std 4, neighborhood baseline 0.5) with goal inhibition, `hold_fixation` and `orienting_saccade`, both on in training.
**Test time**: `exclude_fixation` on (applied to the trained runs; it is not stored in `final_parameters`).

Evidence (mean over seeds; `tests/hparam_exclfix_2026-10-02/exclfix_summary.csv`, `tests/hparam_orient_confirm_2026-10-01/confirm_summary.csv`, `tests/hparam_orient_2026-10-01/orientT_summary.csv`):

| Configuration | Seeds | kNN purity | Pairs sharing | Goals per test | First goals | Goal not executed | Salience saccades | Competence |
|---|---|---|---|---|---|---|---|---|
| `nohold_ior` (earlier reference with inhibition) | 5 | 0.934 | 0.232 | 4.93 | 7.4 | 0.178 | 0.253 | 0.754 |
| `orientT_ior`, test without exclude_fixation | 5 | 0.931 | 0.186 | 5.38 | 13.8 | 0.234 | 0.000 | 0.717 |
| **`orientT_ior`, test with exclude_fixation** (`excl_orientT_ior`) | 5 | **0.938** | 0.200 | 5.41 | 13.8 | **0.000** | **0.000** | n/a |
| `exclfix_orientT_ior` (exclude_fixation also in training) | 5 | 0.872 | 0.207 | 5.48 | 14.4 | 0.000 | 0.000 | 0.719 |

Reasons: it is the only row with purity as high as the reference (0.94), no more goal sharing than the reference, and every goal saccade executed with no salience saccades undoing it (the reference executes 18% of goals not at all and has 25% salience saccades). Training with `exclude_fixation` costs purity (0.87). `nohold_*` runs look good on the recorded scanpaths but are not faithful to the executed saccades. The choice is provisional: it is judged on five seeds, and the report predates these sweeps.

## Parameters (identical for the five seeds; only `init_name` differs)

| Parameter | Value |
|---|---|
| epochs, episodes, saccade_num, saccade_time | 1000, 20, 10, 10 |
| maps_output_size (lattice), maps_learning_rate | 100 (10x10), 0.1 |
| anchor_std, match_std | 4.0, 10.0 |
| neighborhood_modulation, baseline | 40.0, 0.5 |
| learningrate_modulation, baseline | 50.0, 0.02 |
| decaying_speed, local_decaying_speed | 3.0, 0.5 |
| saccade_threshold, agent_sampling_precision | 12.0, 0.999999 |
| predictor_learning_rate, maps_lr_decay | 0.01, 0.0 |
| attention_max_variance, fixed prop, distance prop, slope | 6.0, 0.3, 0.7, 2.0 |
| fovea_scale, fovea_size, fovea_gain, test_fovea | [16, 16], [16, 16], 10000, False |
| retina_scale, taskspace | [80, 80], [0, 80] x [0, 80] |
| gabor scales, orientation bins, kernel size, frequency | [1.0], 5, 5, 0.09 |
| gabor_rgb_prop, gabor_bright_prop | 10.0, 0.0 |
| triangles_percent, colors | 50.0, True |
| random_saccade | "ring" |
| **goal_inhibition, memory, std** | **1.0, 3, 0.5** |
| **hold_fixation** | **True** |
| **orienting_saccade** | **True** |
| exclude_fixation | False in training; **True at test** |

## Runs (five seeds)

- `tests/hparam_orient_2026-10-01/simulations/orientT_ior_{039973,090902,093581}`
- `tests/hparam_orient_confirm_2026-10-01/simulations/confirm_orientT_ior_{027182,031415}`

Current runs in the manifest: 5. More seeds are needed (see below); new runs should use these parameters exactly, through `scripts/grid_search.py --variants`, in a new folder under `tests/`.

## Seed variance and power (Phase 1 spot probe)

Per run, probe values are averaged over the two visual maps (vc, ve). Data: `tests/retinotopy_runs_2026-10-08/retinotopy_runs.csv` (all 107 runs, 32 configurations; configurations with 2 to 5 seeds each).

| Metric | Champion mean (n=5) | Champion SD | Pooled within-configuration SD | SD of configuration means | Between / within |
|---|---|---|---|---|---|
| probe rho | 0.748 | 0.098 | 0.067 | 0.061 | 0.90 |
| probe z | 36.7 | 9.1 | 5.6 | 3.7 | 0.65 |
| probe gain | 0.185 | 0.040 | 0.038 | 0.025 | 0.65 |
| phase-encoded rho | 0.523 | 0.065 | 0.059 | 0.050 | 0.84 |

Reading:
- **Between-configuration variance is no larger than seed variance** (ratio 0.65 to 0.90). With 2 to 5 seeds, configurations cannot be told apart on retinotopy. The choice of champion therefore matters little for Phase 1, and seed noise is the limiting factor. The champion's own SD is the highest of these (0.098 against 0.067 pooled), driven by one seed (039973, rho 0.58).
- **Trained against untrained** is not a power problem: probe rho 0.75 against 0.02 (SD 0.04), Cohen's d about 10. Two seeds are enough to show a difference.
- **Precision of the champion mean** (95% CI half-width on probe rho, from the champion SD 0.098): 0.10 needs 7 seeds, 0.05 needs 18, 0.03 needs 44.
- **Comparing two configurations** (80% power, alpha 0.05, same SD): a difference of 0.10 in probe rho needs 16 seeds per configuration, 0.05 needs 61.
- **"Every map passes" rule** (z > 3 and gain p < 0.05 for both maps of every run): currently 2 of 5 champion runs pass (seeds 031415 and 039973 fail at p of 0.099 and 0.089). All 107 runs: 69% pass. The p value is computed from 100 shuffles (smallest value 0.0099) and is itself noisy. With zero failures, the 95% upper bound on the failure rate is 45% at 5 seeds, 26% at 10, 18% at 15, 14% at 20, 9.5% at 30; the champion already shows a 3/5 failure rate, so a pure "every run passes" rule will not be met at any seed count unless the p test changes (more shuffles, or z only).
- **Practical target**: 15 to 20 seeds (10 to 15 new runs, two batches of 7 at about 90 minutes each; this competes with the 12 retraining runs of predictive_remapping.md inside its 20-run budget) gives a CI half-width of about 0.05 to 0.06 on probe rho. A claim stated as a rate (share of maps with z > 3, which is 100% so far, plus the median and spread of rho and z) is supported at that size. Detecting differences between configurations is not feasible within the 20-run budget.
