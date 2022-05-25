#!/bin/bash
#SBATCH -J BSLM_Bahndanau
#SBATCH -N 1
#SBATCH --mem=16384
#SBATCH --account=HSTR_EinfacheSprache
#SBATCH -t 20:00:00
#SBATCH -o BSLM_Bahndanau2e_5_50EP_PT3-%j.out
#SBATCH -e BSLM_Bahndanau2e_5_50EP_PT3-%j.err
#SBATCH --gres=gpu:V100:1 # select a host with a Volta GPU
#SBATCH --mail-type=END

# Out and err with job id 5606622 is the training iteration of bslm_bahdanau_1e-5_50EP change its name and use accordingly

rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 bert_language_model_test.py

#rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 bleu_test.py