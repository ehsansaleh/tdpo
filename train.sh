#!/bin/bash

mydir=$(pwd) && source ~/.bashrc && cd $mydir

module load openmpi/4.0.5-intel-18.0

conda activate stablebaselines

mpirun -n 36 python train.py --seed 0 --env StandingLegLH-v1
