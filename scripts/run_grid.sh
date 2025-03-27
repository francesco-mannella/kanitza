#!/bin/bash

seeds=1
decay_speeds='6 4'
local_decay_speeds='1.5'
wandb=true
CURR_DIR=$(pwd)
EXE=$(dirname "$0" | xargs realpath | sed -e "s/scripts/src\/main.py/")

params="\
episodes=20;\
epochs=500;\
saccade_num=10;\
saccade_time=10;\
plot_sim=False;\
plot_maps=True;\
plotting_epochs_interval=1;\
agent_sampling_threshold=0.001;\
maps_output_size=100;\
action_size=2;\
attention_size=2;\
maps_learning_rate=0.1;\
saccade_threshold=12.0;\
learningrate_modulation=10.0;\
neighborhood_modulation=10.0;\
learningrate_modulation_baseline=0.02;\
neighborhood_modulation_baseline=0.8;\
match_std_baseline=0.5;\
match_std=8.0;\
anchor_std=2.0;\
triangles_percent=50.0"

for s in $seeds; do
	for ds in $decay_speeds; do
		for lds in $local_decay_speeds; do
			dirname=$(mktemp -d)

			mkdir -p $dirname
			cd $dirname
			if [[ $wandb == false ]]; then wandb disabled; fi

			param_list="${params};decaying_speed=${ds}"
			param_list="${param_list};local_decaying_speed=${lds}"

			(python $EXE --variant='predgrid' --seed=$s --param_list="${param_list}")

			dirname_final=$(cat NAME)
			cd $CURR_DIR

			mv $dirname ./$dirname_final
		done
	done
done
