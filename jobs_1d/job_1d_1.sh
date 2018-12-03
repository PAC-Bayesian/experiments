#!/bin/bash

exp=~/experiments
source activate dev-env
 
# Run the program
python $exp/params/params_1d/params_1.py
python $exp/run_low.py ./params/params_1d/params_1 --num_samples 100 --num_sims 10
python $exp/run_low.py ./params/params_1d/params_1 --num_samples 1000 --num_sims 10
python $exp/run_low.py ./params/params_1d/params_1 --num_samples 5000 --num_sims 5
