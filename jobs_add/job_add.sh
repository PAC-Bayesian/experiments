#!/bin/bash

exp=~/experiments
source activate dev-env
 
# Run the program
python $exp/params/params_add/params_add_15_EI.py
python $exp/run_add.py ./params/params_add/params_add_15_EI --num_samples 500 --num_sims 1 --seed 111