#!/bin/bash

seeds=1
decay_speeds='4'
local_decay_speeds='1.5'
wandb=true
decay_speeds='10 5'
local_decay_speeds=$(seq 2.0 -0.5 1)
wandb=false
CURR_DIR=$(pwd)
EXE=$(dirname "$0" | xargs realpath | sed -e "s/scripts/src\/test.py/" ) 

for s in $seeds; do
    for ds in $decay_speeds; do 
        for lds in $local_decay_speeds; do
            dirname=$(
                printf "s_%04d_ds_%08d_lds_%08d" \
                    "$s" \
                    "$(printf "%.0f" $(echo "$ds * 1000" | bc))" \
                    "$(printf "%.0f" $(echo "$lds * 1000" | bc))"
            )

            mkdir -p ${dirname}_test
            cd ${dirname}_test
            cp ../${dirname}/off_control_store .
            if [[ $wandb == false ]]; then wandb disabled; fi
            python $EXE --seed $s --decaying_speed $ds --local_decaying_speed $lds
            cd $CURR_DIR
        done
    done
done

