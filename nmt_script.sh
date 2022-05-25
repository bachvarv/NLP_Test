#!/bin/bash
#SBATCH -J BNMT
#SBATCH -N 1
#SBATCH --mem=16384
#SBATCH --account=HSTR_EinfacheSprache
#SBATCH -t 30:00:00
#SBATCH -o BNMT-1e-4-%j.out
#SBATCH -e BNMT-1e-4-%j.err
#SBATCH --gres=gpu:V100:1 # select a host with a Volta GPU
#SBATCH --mail-type=END

rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 nmt_model_test.py

#rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 bleu_test.py