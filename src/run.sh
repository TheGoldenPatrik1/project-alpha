#!/bin/sh
#SBATCH --mem-per-cpu=16G
module load python/python3/3.9.6
python python/main.py "$@"
