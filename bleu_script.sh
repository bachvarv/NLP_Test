#!/bin/bash
#SBATCH -J BLEU-Score
#SBATCH -N 1
#SBATCH --mem=16384
#SBATCH --account=HSTR_EinfacheSprache
#SBATCH -t 5:00:00
#SBATCH -o Test_BLEU_both_final-%j.out
#SBATCH -e Test_BLEU_both_final-%j.err
#SBATCH --gres=gpu:V100:1 # select a host with a Volta GPU
#SBATCH --mail-type=END

# Out and err with job id 5606622 is the training iteration of bslm_bahdanau_1e-5_50EP change its name and use accordingly

rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 bleu_test.py