#!/bin/bash
# Run the test scripts on every trained simulation folder.
#
# Usage: bash /path/to/scripts/tests.sh [-g] [GLOB] [EXTRA_ARGS...]
#     -g: run src/test_generative.py (no plots) instead of
#         src/test.py --plot; each folder then also needs rnn_store.npy.
#     GLOB: folders to test (default "*_s_*_m_*", i.e. run_grid.sh outputs);
#         give it explicitly when passing EXTRA_ARGS.
#     EXTRA_ARGS: forwarded to the test script, e.g.
#         --mask_posrot 40 40 0 --mask_start 20 --arbitration
#
# In each matching folder runs, for shape in {triangle, square} and rot in
# seq 0 0.2 1.6 (radians),
#     <test script> --posrot 40 40 <rot> --world <shape> --skip_existing
# so single tests whose goals file already exists are skipped.
# Outputs (per folder): goals-<shape>-40-0-40-0-<rot>[-<w>].npy, plus
# sim/maps/merged test gifs and pngs without -g.
set -e

SRC_DIR="$(dirname "$(realpath "$0")")/../src"
TEST_APP="$SRC_DIR/test.py"
TEST_ARGS=(--plot)
if [[ $1 == -g ]]; then
    TEST_APP="$SRC_DIR/test_generative.py"
    TEST_ARGS=()
    shift
fi
search_dir=${1:-*_s_*_m_*}
shift || true
EXTRA_ARGS=("$@")

INITIAL_DIR=$(pwd)

for EXPERIMENT_DIR in $search_dir; do
    if [ -d "$EXPERIMENT_DIR" ]; then
        echo "Testing on $EXPERIMENT_DIR ..."
        cd "$EXPERIMENT_DIR"
        for SHAPE in triangle square; do
            for ROTATION in $(seq 0 0.2 1.6); do
                python "$TEST_APP" "${TEST_ARGS[@]}" \
                    --posrot 40 40 "$ROTATION" --world "$SHAPE" \
                    --skip_existing "${EXTRA_ARGS[@]}"
            done
        done
        cd "$INITIAL_DIR"
    fi
done
