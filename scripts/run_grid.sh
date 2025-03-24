#!/bin/bash

seeds=1
decay_speeds='4'
local_decay_speeds='1.5'
match_std='8.0 4.0 3.0'
neighborhood_modulation='8.0 5.0 3.0'
wandb=true
CURR_DIR=$(pwd)
EXE=$(dirname "$0" | xargs realpath | sed -e "s/scripts/src\/main.py/" ) 

for s in $seeds; do
    for ms in $match_std; do
        for nm in $neighborhood_modulation; do
            for ds in $decay_speeds; do 
                for lds in $local_decay_speeds; do
                    dirname=$(mktemp -d)

                    mkdir -p $dirname
                    cd $dirname
                    if [[ $wandb == false ]]; then wandb disabled; fi

                    param_list="decaying_speed=${ds}"
                    param_list="${param_list};local_decaying_speed=${lds}"
                    param_list="${param_list};match_std=${ms}"
                    param_list="${param_list};neighborhood_modulation=${nm}"

                    ( python $EXE --variant='colgrid' --seed=$s --param_list="${param_list}" )

                    dirname_final=$(cat NAME)
                    cd $CURR_DIR

                    mv $dirname ./$dirname_final
                done
            done
        done
    done
done

