#!/bin/sh
#SBATCH --mem-per-cpu=8G
module load python/python3/3.9.6
python script.py "$@"
