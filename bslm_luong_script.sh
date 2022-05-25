#!/bin/bash
#SBATCH -J BSLM_Luong
#SBATCH -N 1
#SBATCH --mem=16384
#SBATCH --account=HSTR_EinfacheSprache
#SBATCH -t 20:00:00
#SBATCH -o BSLM_Luong2e_5_50PT3-%j.out
#SBATCH -e BSLM_Luong2e_5-50PT3-%j.err
#SBATCH --gres=gpu:V100:1 # select a host with a Volta GPU
#SBATCH --mail-type=END

rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 bslm_luong_test.py

#rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 bleu_test.py