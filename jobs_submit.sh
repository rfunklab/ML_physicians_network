#!/bin/bash
# 1: algorithm: en, lasso, rfr
# 2: pixel resolution: small, mid, large
# 3: random seed


case ${1} in
 en)
 time="96:00:00"
 ;;
 
 rfr)
 time="72:00:00"
 ;;
 
 lasso)
 time="10:00:00"
 ;;
 
 *)
 echo "${1} is not a recognized option. Plesase use rfr, lasso, or en."
 exit 1
 ;;

esac

echo Submitting job...
echo sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-0_seed-${3} --export=alg=${1},resolution=${2},hdim=0,seed=${3} --time=${time} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-0_seed-${3} --export=alg=${1},resolution=${2},hdim=0,seed=${3} --time=${time} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
echo 

echo Submitting job...
echo sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-1_seed-${3} --export=alg=${1},resolution=${2},hdim=1,seed=${3} --time=${time} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
sbatch --job-name=cv_alg-${1}_outcome-ar_stdprice_total_res-${2}_hdim-1_seed-${3} --export=alg=${1},resolution=${2},hdim=1,seed=${3} --time=${time} --mail-user=fpichard@umn.edu submit_scripts/cv_outcome-ar_stdprice_total.sbatch
echo
