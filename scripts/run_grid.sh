#!/bin/bash
# Launch a grid of training simulations (src/main.py).
#
# Usage: run from the directory that will hold the simulation folders:
#     bash /path/to/scripts/run_grid.sh
#
# For each combination of seeds x decay_speeds x local_decay_speeds x
# agent_sampling_precision, builds an id
#     <series>_s_<seed>_m_<match_std>_a_<anchor_std>_d_<ds>_l_<lds>_p_<precision>
# (numbers zero-padded to 6 chars, dot removed) and, unless a folder with that
# id already exists in the cwd, runs main.py inside a mktemp dir with
# --variant=<id> --seed=<seed> --param_list="<params>;decaying_speed=..."
# and finally moves the temp dir to ./<content of NAME>. If main.py fails the
# temp dir is left in place and its path is printed.
#
# Inputs: variables at the top of this file (grid values, `params` string).
# Outputs: one folder per simulation containing loaded_params, log, NAME,
#     off_control_store, maps_*.gif/png (see SCRIPTS.md).
# `wandb=true` passes -w to main.py (log to wandb); with `wandb=false` main.py
# logs to <run>/data_sim instead.

seeds="1"
# decay_speeds="3.5"
# local_decay_speeds="0.5"
# wandb=false
decay_speeds="3.0"
local_decay_speeds="1.0"
agent_sampling_precision="0.999"
wandb=true
match_std="8.0"
anchor_std="2.0"
series=noise_sampled
CURR_DIR=$(pwd)
CURR_SIMS=$(ls | grep $series)
EXE="$(dirname "$(realpath "$0")")/../src/main.py"

fmt() { printf "%06.3f" "$1" | sed -e "s/\.//"; }

params="\
episodes=20;\
epochs=500;\
saccade_num=10;\
saccade_time=10;\
plot_sim=False;\
plot_maps=True;\
plotting_epochs_interval=100;\
maps_output_size=100;\
action_size=2;\
attention_size=2;\
maps_learning_rate=0.1;\
saccade_threshold=12.0;\
attention_max_variance=1.0;\
learningrate_modulation=10.0;\
neighborhood_modulation=20.0;\
learningrate_modulation_baseline=0.02;\
neighborhood_modulation_baseline=0.8;\
match_std_baseline=0.5;\
match_std=${match_std};\
anchor_std=${anchor_std};\
triangles_percent=50.0;\
colors=True"

for s in $seeds; do
	for ds in $decay_speeds; do
		for lds in $local_decay_speeds; do
            for precision in $agent_sampling_precision; do

                id_="${series}_s_${s}_m_$(fmt $match_std)_a_$(fmt $anchor_std)"
                id_="${id_}_d_$(fmt $ds)_l_$(fmt $lds)_p_$(fmt $precision)"

                sim_exists=false
                [[ $CURR_SIMS =~ $id_ ]] && sim_exists=true

                if [[ $sim_exists == true ]]; then
                    echo "$id_ exists. Simulation not started."
                else
                    echo  "$id_ does not exists, simulating..."
            
                    dirname=$(mktemp -d)
                    #
                    mkdir -p $dirname
                    cd $dirname
                    wandb_flag=""
                    if [[ $wandb == true ]]; then wandb_flag="-w"; fi
                    #
                    param_list="${params};decaying_speed=${ds}"
                    param_list="${param_list};local_decaying_speed=${lds}"
                    param_list="${param_list};agent_sampling_precision=${precision}"
                    #
                    if python "$EXE" --variant=$id_ --seed=$s \
                        --param_list="${param_list}" $wandb_flag \
                        && [[ -s NAME ]]; then
                        dirname_final=$(cat NAME)
                        cd "$CURR_DIR"
                        mv "$dirname" "./$dirname_final"
                    else
                        echo "$id_ failed, output left in $dirname"
                        cd "$CURR_DIR"
                    fi
                fi
            done
		done
	done
done
