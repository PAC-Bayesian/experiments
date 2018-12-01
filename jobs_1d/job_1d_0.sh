#!/bin/bash

source activate dev-env
 
# Run the program
python ./params/params_1d/params_0.py
python ./run_1d.py ./params/params_1d/params_0 --num_samples 100 --num_sims 20
python ./run_1d.py ./params/params_1d/params_0 --num_samples 1000 --num_sims 20
python ./run_1d.py ./params/params_1d/params_0 --num_samples 10000 --num_sims 20
