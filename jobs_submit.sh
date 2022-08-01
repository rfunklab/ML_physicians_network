#!/bin/bash
# 1: algorithm: en, lasso, rfr
# 2: pixel resolution: small, mid, large
# 3: random seed

echo Submitting job...
echo sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-0_seed-${3} --export=alg=${1},resolution=${2},hdim=0,seed=${3} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-0_seed-${3} --export=alg=${1},resolution=${2},hdim=0,seed=${3} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
echo 

echo Submitting job...
echo sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-1_seed-${3} --export=alg=${1},resolution=${2},hdim=1,seed=${3} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-1_seed-${3} --export=alg=${1},resolution=${2},hdim=1,seed=${3} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
echo
