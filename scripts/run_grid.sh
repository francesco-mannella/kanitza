#!/bin/bash


#####
# This script configures and runs simulations with customizable parameters such as seeds, decay speeds, and local decay speeds. It also includes an option to enable or disable Weights & Biases (wandb) integration.

# Default values
seeds="1"
decay_speeds="3.5"
local_decay_speeds="0.5"
wandb=false
series="big_field_slope"

#####
# Parse options
OPTIONS=s:d:l:wr:h
LONGOPTIONS=seeds:,decay-speeds:,local-decay-speeds:,wandb,series:,help

# Use getopt for parsing
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    exit 2
fi
eval set -- "$PARSED"

# Process options
while true; do
    case "$1" in
        -s|--seeds)
            seeds="$2"
            shift 2
            ;;
        -d|--decay-speeds)
            decay_speeds="$2"
            shift 2
            ;;
        -l|--local-decay-speeds)
            local_decay_speeds="$2"
            shift 2
            ;;
        -w|--wandb)
            wandb=true
            shift
            ;;
        -r|--series)
            series="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -s, --seeds                Set the seeds value (default: 1)"
            echo "  -d, --decay-speeds         Set the decay speeds value (default: 3.5)"
            echo "  -l, --local-decay-speeds   Set the local decay speeds value (default: 0.5)"
            echo "  -w, --wandb                Enable Weights & Biases integration (default: false)"
            echo "  -r, --series               Set the series value (default: big_field_slope)"
            echo "  -h, --help                 Display this help message"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done
#####

CURR_DIR=$(pwd)
CURR_SIMS=$(ls | grep sim_ | grep $series)
EXE=$(dirname "$0" | xargs realpath | sed -e "s/scripts/src\/main.py/")

params="\
episodes=20;\
epochs=500;\
saccade_num=10;\
saccade_time=10;\
plot_sim=False;\
plot_maps=True;\
plotting_epochs_interval=1;\
agent_sampling_threshold=0.00001;\
maps_output_size=100;\
action_size=2;\
attention_size=2;\
maps_learning_rate=0.1;\
learningrate_modulation=10.0;\
neighborhood_modulation=20.0;\
learningrate_modulation_baseline=0.02;\
neighborhood_modulation_baseline=0.8;\
match_std_baseline=0.5;\
match_std=8.0;\
anchor_std=2.0;\
triangles_percent=50.0;\
colors=True"

for s in $seeds; do
	for ds in $decay_speeds; do
		for lds in $local_decay_speeds; do

			id_="s_${s}_m_08000_a_02000_d_$(echo $ds | xargs printf "%06.3f" | sed -e "s/\.//")"
			id_="${id_}_l_$(echo $lds | xargs printf "%06.3f" | sed -e "s/\.//")"

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

                USE_WANDB=-w
				if [[ $wandb == false ]]; then USE_WANDB=; fi
				
				param_list="${params};decaying_speed=${ds}"
				param_list="${param_list};local_decaying_speed=${lds}"
				#
                (python -u $EXE --variant=$series $USE_WANDB --seed=$s --param_list="${param_list}" 2>&1 | tee errlog )
				#
				dirname_final=$(cat NAME)
				cd $CURR_DIR
				#
				mv $dirname ./$dirname_final
			fi
		done
	done
done
