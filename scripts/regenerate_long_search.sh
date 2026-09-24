#!/bin/bash
# Regenerate the training data of tests/long_search_b71233_090902 (and,
# optionally, of its sibling long_search_57a9b5_090902) with the current
# code of this branch.
#
# Usage: bash /path/to/scripts/regenerate_long_search.sh OUTDIR [--sibling] [-w]
#     OUTDIR: directory where the run folders are created.
#     --sibling: also regenerate long_search_57a9b5_090902 (same parameters
#         with local_decaying_speed=1.0 instead of 0.5), in parallel.
#     -w: log to wandb, as the original runs did.
#
# Each run folder gets the files main.py writes: loaded_params, log, NAME,
# off_control_store, maps_<epoch>.gif/png every 100 epochs up to epoch 999,
# plus nohup.out with the console output. The parameters and seed are the
# ones recorded in the original runs' wandb-metadata.json (launched by
# src/grid_search.py at commit a1fbe79). If a run folder already contains an
# off_control_store, main.py resumes it up to 1000 epochs in total.
#
# The results will not match the original runs exactly: the code changed
# after a1fbe79 (Gabor filters, agent, parameters), and the later fixes
# change the controller's visual input during training, the competence
# predictor and the match scores (see SCRIPTS.md).
set -e

usage() {
    sed -n '6,10p' "$0" | sed -e 's/^# \{0,1\}//'
}

OUTDIR=""
SIBLING=false
WANDB_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sibling) SIBLING=true ;;
        -w) WANDB_FLAG="-w" ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option $1"; usage; exit 1 ;;
        *) OUTDIR=$1 ;;
    esac
    shift
done
if [[ -z $OUTDIR ]]; then usage; exit 1; fi

MAIN="$(dirname "$(realpath "$0")")/../src/main.py"
SEED=90902

params() {
    echo "test_fovea=False;episodes=20;epochs=1000;saccade_num=10;\
saccade_time=10;plot_sim=False;plot_maps=True;plotting_epochs_interval=100;\
maps_output_size=100;action_size=2;attention_size=2;maps_learning_rate=0.1;\
saccade_threshold=12.0;decaying_speed=3.0;local_decaying_speed=$1;\
learningrate_modulation=50.0;neighborhood_modulation=40.0;\
learningrate_modulation_baseline=0.02;neighborhood_modulation_baseline=0.1;\
match_std_baseline=0.5;match_std=10.0;anchor_std=2.0;triangles_percent=50.0;\
agent_sampling_precision=0.999999;gabor_scales=[1.0];gabor_orientation_bins=5;\
gabor_frequency=0.09;gabor_sigma_y_multiplier=1;gabor_kernel_size=5;\
gabor_phase_offset=-1.4828317324943823;gabor_rgb_prop=10.0;gabor_bright_prop=0.0;\
attention_max_variance=6;attention_fixed_variance_prop=0.3;\
attention_center_distance_variance_prop=0.7;attention_center_distance_slope=2;\
fovea_scale=[16, 16];fovea_size=[16, 16]"
}

run() {
    local name=$1 local_decaying_speed=$2
    mkdir -p "$OUTDIR/$name"
    echo "Running $name in $OUTDIR/$name (output in nohup.out) ..."
    (
        cd "$OUTDIR/$name"
        python -u "$MAIN" -r "$name" -p "$(params "$local_decaying_speed")" \
            -s $SEED $WANDB_FLAG > nohup.out 2>&1
    )
}

# Stop the background runs too on Ctrl+C or kill
trap 'kill 0' INT TERM

run long_search_b71233_090902 0.5 &
if [[ $SIBLING == true ]]; then
    run long_search_57a9b5_090902 1.0 &
fi
wait
echo "Done."
