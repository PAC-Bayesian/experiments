#!/bin/bash

exp=~/experiments
source activate dev-env
 
# Run the program
python $exp/params/params_1d/params_3.py
python $exp/run_1d.py ./params/params_1d/params_3 --num_samples 100 --num_sims 20
python $exp/run_1d.py ./params/params_1d/params_3 --num_samples 1000 --num_sims 20
python $exp/run_1d.py ./params/params_1d/params_3 --num_samples 10000 --num_sims 20
