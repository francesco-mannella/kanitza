#!/bin/bash

seeds="1"
# decay_speeds="3.5"
# local_decay_speeds="0.5"
# wandb=false
decay_speeds="3.0 3.5 4.0"
local_decay_speeds="0.5 1.0"
wandb=true
series=polar_saccade_reset
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
plotting_epochs_interval=100;\
agent_sampling_threshold=0.00001;\
maps_output_size=100;\
action_size=2;\
attention_size=2;\
maps_learning_rate=0.1;\
saccade_threshold=12.0;\
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
				if [[ $wandb == false ]]; then wandb disabled; fi
				#
				param_list="${params};decaying_speed=${ds}"
				param_list="${param_list};local_decaying_speed=${lds}"
				#
				(python $EXE --variant=$series --seed=$s --param_list="${param_list}")
				#
				dirname_final=$(cat NAME)
				cd $CURR_DIR
				#
				mv $dirname ./$dirname_final
			fi
		done
	done
done
