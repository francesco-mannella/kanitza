#!/bin/bash

set -e

TEST_APP=${HOME}/Projects/current/kanizsa_tests/src/test.py
# Store the initial working directory
INITIAL_DIR=$(pwd)

# Iterate through directories matching the pattern "s_1*00"
#for EXPERIMENT_DIR in s_1*00; do
for EXPERIMENT_DIR in sim_polar*; do
    echo $EXPERIMENT_DIR
	# Check if the directory contains a file named "goal"
	if [[ -z "$(ls "$EXPERIMENT_DIR" | grep goal)" ]]; then
		# Change the current directory to the experiment directory
		cd "$EXPERIMENT_DIR"

		# Iterate through shapes (triangle and square)
		for SHAPE in triangle square; do
			# Iterate through rotation values from 0 to 1 with a step of 0.2
			for ROTATION in $(seq 0 0.2 1.6); do

				# Disable wandb (Weights & Biases)
				wandb disabled

				# Execute the Python script with specified parameters
				python ${TEST_APP} --posrot 40 40 "$ROTATION" --world "$SHAPE"
			done
		done

		# Change the current directory back to the initial directory
		cd "$INITIAL_DIR"
	fi
done
