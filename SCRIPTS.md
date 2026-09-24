# Scripts: use cases and known issues (saliencies branch)

## Setup

- Install the environment: `pip install -e tools/EyeSim` (gymnasium, box2d-py, numpy, matplotlib, scikit-image). Use the editable `-e` install: a plain `pip install` copies the package into site-packages, and edits to `tools/EyeSim` are then silently ignored.
- Other dependencies: torch, opencv (`opencv-python-headless`, used by the agent to resize the fovea), wandb, pandas, seaborn, python-slugify, Pillow, svg.path (only for `svg2json.py`).
- Every `src` script imports `model.*`, `params` and `plotter` as top-level modules. Run them either as `python /path/to/src/<script>.py` (Python adds `src/` to the path itself) or with `PYTHONPATH=/path/to/src`.
- The entry scripts read and write files in the **current working directory**. `cd` into the simulation folder first.

## Pipeline

```
src/grid_search.py ───> src/main.py        (one folder per simulation: training;
scripts/run_grid.sh ──┘                     run_grid.sh is the older launcher)
                           │  off_control_store, loaded_params, log, NAME, maps_*.gif/png
                           v
scripts/tests.sh ─────> src/test.py        (in each folder: test scanpaths)
                           │  goals-<world>-<pos>-<rot>.npy (+ gifs with --plot)
                           v
scripts/paths.py ─────> paths.csv, <sim>.png
                           v
src/render_scanpaths.py / render_scanpaths_alt.py   (animations)

Side branches:
  scripts/tests.sh -g ─> src/test_generative.py  (needs rnn_store.npy)
  src/test_weights.py           (inspect map prototypes)
  src/analysis.py, src/wdb_analysis.py   (parameter-sweep plots)
  src/demos/*.py                (agent/attention demos, no controller)
```

## scripts/

| Script | Use case | Run from | Reads | Writes |
|---|---|---|---|---|
| `run_grid.sh` | Older launcher, superseded by `src/grid_search.py`: its `params` string has no Gabor or fovea parameters, so those take their defaults. Train a grid of simulations. Grid values and the base `params` string are set at the top of the file; `wandb=true` passes `-w` to `main.py`. It skips ids that already exist in the cwd. Ids look like `<series>_s_<seed>_m_<match_std>_a_<anchor_std>_d_<ds>_l_<lds>_p_<prec>`, built from the same variables that go into `params`. If `main.py` fails, the temp dir is left in place and its path is printed. | the output dir | none | one folder per simulation (see `main.py`) |
| `tests.sh [-g] [GLOB] [EXTRA_ARGS...]` | Run `test.py --plot` (or, with `-g`, `test_generative.py` without plots) on every folder matching GLOB (default `*_s_*_m_*`), for triangle and square, rotation `seq 0 0.2 1.6` rad, position (40, 40). Single tests whose goals file already exists are skipped (`--skip_existing`). `EXTRA_ARGS` are forwarded, for example `tests.sh -g '*_s_*_m_*' --mask_posrot 40 40 0 --mask_start 20 --arbitration`. | the sims dir | `off_control_store`, `loaded_params` (+ `rnn_store.npy` with `-g`) | `goals-*.npy`, `*_test_*.gif/png` |
| `regenerate_long_search.sh OUTDIR [--sibling] [-w]` | Retrain `tests/long_search_b71233_090902` (and, with `--sibling`, `long_search_57a9b5_090902` in parallel) with the current code, the recorded parameters and seed 90902. Results will differ from the originals (see the script header). | any | none | `OUTDIR/<run>/` with the `main.py` outputs and `nohup.out` |
| `paths.py` | Collect all `goals*.npy` into one table (including `precision`, parsed from `_p_` in the sim name), dropping the first 3 saccades of each test, and plot goal trajectories per simulation. | the sims dir | `*_s_*_m_*/goals*.npy` | `paths.csv`, `<sim>.png` |

## src/ entry points

### `grid_search.py`: parameter grids (produced `tests/long_search_*`)
`python src/grid_search.py`, run from the output directory. The seeds, `params` (a list value means grid over its alternatives, any other value, strings included, is fixed; list-valued parameters are double-wrapped, for example `gabor_scales=[[1.0]]`), `base_name`, `WANDB` and `MAX_PROCESSES` are constants in the file. For each combination and seed it creates `<base_name>_<md5(params)[:6]>_<seed:06d>/` and runs `nohup python -u main.py -r <name> -p '<k=v;...>' -s <seed> [-w]` inside it.

### `main.py`: training
`python src/main.py [-s N] [-r NAME] [-p "k=v;..."] [-w] [-o]`
- The first run in a folder merges `-p` into the defaults in `params.py` and saves the result to `loaded_params`. Later runs load `loaded_params`; a `-p` given then is applied on top, the changed keys are printed, and the file is re-saved.
- Values in `-p` and `loaded_params` are Python literals (`3.0`, `True`, `'text'`, `[0, 80]`); unquoted text is taken as a string. A value whose type does not match the parameter's default raises `TypeError` (int and float are interchangeable).
- If `off_control_store` exists, training resumes from it and stops once `epochs` epochs have been run in total.
- Each epoch runs `episodes` × `saccade_num` × `saccade_time` steps. The agent filters the retina with the Gabor bank (`gabor_*` parameters), samples a salient point inside its attentional mask and moves the eye there. The maps' visual input is the agent's color-saliency fovea: the central `fovea_scale` retina pixels, resized to `fovea_size` and multiplied by `fovea_gain` (1e4), or the raw FOVEA if `test_fovea`. At the midpoint of each saccade the controller proposes an attention center from the same saliency fovea (`Agent.get_fovea`). Afterwards the salient saccades (action norm > `saccade_threshold`) update the three maps and the competence predictor. See `pseudocode.md`.
- `-w` and `-o` are runtime flags: they always come from the command line, never from `loaded_params`.
- `-o` shows the fovea plotter live.
- Outputs: `NAME`, `loaded_params`, `log`, `off_control_store` (saved every epoch), `maps_<epoch>.gif/png`, and `sim_<epoch>.gif` if `plot_sim`. With `-w`, wandb logs competence, weight changes and the maps.

### `test.py`: test scanpaths
`python src/test.py [--plot] [--seed N] [--world triangle|square] [--posrot X Y A] [--wandb] [--skip_existing]`
- Exits with an error if `off_control_store` is missing.
- Runs 1 episode of 16 × `saccade_time` steps. Every `saccade_period` (4) steps the fovea picks a winner on the visual-conditions map, and the matching attention-map weight becomes the attention center.
- Saves `goals-<world>-<pos>-<rot>.npy` (slugified; `--skip_existing` makes it do nothing if that file exists), a list holding one dict with the keys `world`, `position`, `angle`, `saccade_id` (`<episode>-<ts>`) and `goal` ((1,2) map point). With `--plot` it also saves the `sim_test_*`, `maps_test_*` and `merged_test_*` gifs.

### `test_generative.py`: test with RNN goal arbitration
`python src/test_generative.py [...test.py args] [--mask_type T] [--mask_posrot X Y A] [--mask_start T] [--arbitration]`
- Same loop as `test.py`, but first feeds the unfiltered map goal to the FORCE RNN (`rnn_store.npy`). The RNN prediction then filters the visual-conditions map before the final goal is chosen. From `mask_start - 4` on, `--arbitration` sets the RNN weight to 1.
- `rnn_store.npy` is not produced by any script in this repo (see issue #11).
- An occluder (`*_mask` bodies in `worlds.json`) can be placed with `--mask_posrot`; it appears at `--mask_start`.
- The saved dicts also contain `offcontrol_goal` and `rnn_goal` (numpy arrays); the file name ends with the arbitration weight.

### `test_weights.py`: inspect map prototypes
`python src/test_weights.py` inside a trained folder. It saves `visual_weights.png`, a tiling of the normalized visual-conditions prototypes on the map grid (shapes from `loaded_params`).

### Analysis and rendering
- `analysis.py`: scatter of final competence over (`decaying_speed`, `local_decaying_speed`) for the `*_s_*_m_*` folders. It needs the `comp:` lines in `log`, which `main.py` writes only with `-w`.
- `wdb_analysis.py`: downloads the `predgrid` runs from wandb (cached to `stats.csv`) and plots competence and smoothed weight change over the decay grid, saved to `parameter_exploration.png`.
- `render_scanpaths.py`: animates the trajectory of each trial on a 7×7 grid, from `paths.csv` (sims with precision 0.7, first 6 saccades of each trial skipped).
- `render_scanpaths_alt.py`: the same, drawn as Gaussian blobs on a 10×10 grid (precision 0.8).
- `merge_gifs.py`: library function that stacks two frame lists into one gif (used by the test scripts).

### Demos (`src/demos/`, interactive; each saves `episode_000{0,1,2}.gif/png` in the cwd)
Run them with `PYTHONPATH=/path/to/src`.

- `attentional_demo.py`: a ring of 15 fixed attention centers per object; shows how the attentional mask steers saliency sampling.
- `attentional_sequence_demo.py`: replays the recorded displacements in `retina_poses.npy`; after the first 3 saccades the retina is blanked, so saccades follow attention only. Run it from `src/demos`.
- `random_demo.py`: pure bottom-up saliency with a uniform mask and a low sampling threshold (0.01).

## Library modules

| Module | Content |
|---|---|
| `params.py` | `Parameters`: every hyperparameter, documented in the class docstring. |
| `parameter_manager.py` | Parsing of `"k=v;..."` strings, `save`/`load` of `loaded_params`. |
| `model/agent.py` | Gaussian attentional mask and thresholded sampling over the Gabor saliency, which together produce the retina action and the color-saliency fovea. |
| `model/visual_processing.py` | `SaliencyMap`: builds the Gabor bank from the `gabor_*` parameters. |
| `model/gabor_filtering.py` | `gabor_kernel` and `ChannelGaborFilter` (red/green/blue opponent and brightness channels); runnable demo, `python gabor_filtering.py [IMAGE_PATH_OR_URL ...]` (default `src/model/gabor_test.png`, untracked). |
| `model/offline_controller.py` | The three maps, state storage, offline update, competence, save/load. |
| `model/topological_maps.py` | `TopologicalMap`/`FilteredTopologicalMap` (torch), the `Updater` (SOM/STM loss). |
| `model/predict.py` | Logistic competence predictor and its updater. |
| `model/recurrent_generative_model.py` | FORCE reservoir that encodes and anticipates map goals. |
| `params_recurrent_generative_model.py` | `ParamsFORCE`. |
| `plotter.py` | `FoveaPlotter` (scene, fovea, saliency, mask) and `MapsPlotter` (the three maps). |
| `tools/EyeSim` | Gymnasium env `EyeSim/EyeSim-v0`, built with `gym.make(..., params=Parameters)`: scene, retina and fovea sizes come from `taskspace_*`, `retina_*` and `fovea_*` (default 80×80 scene and retina, 16×16 px FOVEA observation); colored objects from `models/worlds.json`. `models/svg2json.py` builds world JSONs from SVG. |

## Known limitations

| # | Location | Note |
|---|---|---|
| 11 | `src/model/recurrent_generative_model.py` | No script in this repo trains the RNN; `rnn_store.npy` must be produced elsewhere before running `test_generative.py`. |
| 14 | `main.py` | The world is chosen per epoch, so every update sees a single class (block curriculum, by design). |

## Changes that affect comparison with earlier runs

- Competence predictor: each sample is now trained on its own match. Before, a shape bug trained every sample toward the batch mean.
- Match score: only the attention and visual-effects winners are compared with the goal. The old constant third term, which put a floor of 1/3 under every match, is gone.
- `test_generative.py`: the unfiltered goal no longer inherits the RNN filter from the previous saccade.
- `paths.csv`: the first 3 saccades of each test are dropped. Before, only timestep 0 was.
- `main.py` resume: `epochs` is now the total number of epochs, not the number of additional ones.
- Visual input of the controller: `main.py`, `test.py` and `test_generative.py` now query the maps with the agent's color-saliency fovea, the same input the maps are trained on. Before, `generate_saccade` in training got the raw FOVEA, `test.py` the Gabor saliency of the FOVEA and `test_generative.py` the raw FOVEA. This affects training itself, so the `long_search` runs are not reproducible with the current code.
- Gabor orientations: `linspace(0, 180, gabor_orientation_bins)[:-1]` instead of `linspace(0, 360, ...)[:-1]`. With `bins=5`, as in `long_search`, that is 0/45/90/135 instead of 0/90/180/270. On 30 test scenes the saliency maps correlate at 0.98 with the old ones, but the saliency peak moves by 9 px on average (up to 28 px).
- `Parameters.fovea_scale` now defaults to [16, 16] (was [50, 50]), matching the runs. Runs that relied on the default get a different fovea. The env FOVEA observation now honors `fovea_scale`; it is unchanged when `fovea_scale == fovea_size`.
- `test.py`/`test_generative.py` `--posrot`: the angle is passed to the env in radians as given. Before, it was converted to degrees and then used as radians. Goals file names now carry the given angle (for example `-000-40` for 0.4).
