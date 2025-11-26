#!/bin/bash

#####
# This script configures and runs simulations with customizable parameters such as seeds, decay speeds, and local decay speeds. It also includes an option to enable or disable Weights & Biases (wandb) integration.

# Default values
seeds="1"
# decay_speeds="3.5"
# local_decay_speeds="0.5"
# wandb=false
decay_speeds="3.0"
local_decay_speeds="1.0"
agent_sampling_precision="0.999"
wandb=true
series=noise_sampled
CURR_DIR=$(pwd)
CURR_SIMS=$(ls | grep $series)
EXE=$(dirname "$0" | xargs realpath | sed -e "s/scripts/src\/main.py/")

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
match_std=8.0;\
anchor_std=2.0;\
triangles_percent=50.0;\
colors=True"

for s in $seeds; do
	for ds in $decay_speeds; do
		for lds in $local_decay_speeds; do
			for precision in $agent_sampling_precision; do

				id_="${series}_s_${s}_m_08000_a_02000_d_$(echo $ds | xargs printf "%06.3f" | sed -e "s/\.//")"
				id_="${id_}_l_$(echo $lds | xargs printf "%06.3f" | sed -e "s/\.//")"
				id_="${id_}_p_$(echo $precision | xargs printf "%06.3f" | sed -e "s/\.//")"

				dirname=$(mktemp -d)
				#
				mkdir -p $dirname
				cd $dirname

				if [[ $sim_exists == true ]]; then
					echo "$id_ exists. Simulation not started."
				else
					echo "$id_ does not exists, simulating..."

					dirname=$(mktemp -d)
					#
					mkdir -p $dirname
					cd $dirname
					if [[ $wandb == false ]]; then wandb disabled; fi
					#
					param_list="${params};decaying_speed=${ds}"
					param_list="${param_list};local_decaying_speed=${lds}"
					param_list="${param_list};agent_sampling_precision=${precision}"
					#
					(python $EXE --variant=$id_ --seed=$s --param_list="${param_list}")
					#
					dirname_final=$(cat NAME)
					cd $CURR_DIR
					#
					mv $dirname ./$dirname_final
				fi
			done
		done
	done
done
