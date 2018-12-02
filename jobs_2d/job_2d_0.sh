#!/bin/bash

exp=~/experiments
source activate dev-env
 
# Run the program
python $exp/params/params_2d/params_0.py
python $exp/run_low.py ./params/params_2d/params_0 --num_samples 200 --num_sims 20
python $exp/run_low.py ./params/params_2d/params_0 --num_samples 2000 --num_sims 20
python $exp/run_low.py ./params/params_2d/params_0 --num_samples 20000 --num_sims 10