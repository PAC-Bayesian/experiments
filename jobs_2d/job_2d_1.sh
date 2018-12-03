#!/bin/bash

exp=~/experiments
source activate dev-env
 
# Run the program
python $exp/params/params_2d/params_1.py
python $exp/run_low.py ./params/params_2d/params_1 --num_samples 400 --num_sims 10
python $exp/run_low.py ./params/params_2d/params_1 --num_samples 2000 --num_sims 10
python $exp/run_low.py ./params/params_2d/params_1 --num_samples 10000 --num_sims 5