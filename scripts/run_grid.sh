#!/bin/bash

seeds=1
decay_speeds='5 4 3'
local_decay_speeds='1.5 1.3 1.1'
match_std='4.0 3.5 3.0'
wandb=true
CURR_DIR=$(pwd)
EXE=$(dirname "$0" | xargs realpath | sed -e "s/scripts/src\/main.py/" ) 

for s in $seeds; do
    for ms in $match_std; do
        for ds in $decay_speeds; do 
            for lds in $local_decay_speeds; do
                dirname=$(mktemp -d)

                mkdir -p $dirname
                cd $dirname
                if [[ $wandb == false ]]; then wandb disabled; fi

                param_list="decaying_speed=${ds}"
                param_list="${param_list};local_decaying_speed=${lds}"
                param_list="${param_list};match_std=${ms}"

                ( python $EXE --variant='filter_grid' --seed=$s --param_list="${param_list}" )

                dirname_final=$(cat NAME)
                cd $CURR_DIR

                mv $dirname ./$dirname_final
            done
        done
    done
done

