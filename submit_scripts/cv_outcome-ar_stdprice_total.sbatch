#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --ntasks=8
#SBATCH --mem=10g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --output=../log_files/SB-%x.%j.out
#SBATCH --error=../log_files/SB-%x.%j.err
####
#
#Pass a path to a file for a script you are running.
#Use --export=input_file=PATH_TO_FILE,OTHER_VAR=VALUE
#Use --job-name=NAME to distinguish the sbatch job in the queue.
#Use --mail-user=EMAIL@EMAIL.EDU
#Use -p small to submit to small queue
#Use ""sbatch {Options} -p small SCRIPT.sbtach" to submit job.
#
# Example:
#   sbatch --job-name=IDX_NAME_OF_JOB --mail-user=YOUR@EMAIL.EDU --export=input_file=../path,cohort_name=mf,ses=06 -p small submit_file.sbatch
#
#Created: FMP
#Last updated: 17032021 - FMP
####

##Move to pipeline dir
cd ${SLURM_SUBMIT_DIR}


###LOAD NECESSARY MODULES!
## Needs to go before loading defaults for globus dirs to work
#module load fsl/6.0.1
module load python3
#module load camino
#module load freesurfer/6.0.0
#module load globus-cli
#module load afni/19.3.14
#module load ants
#module load c3d
#module load rstudio
#module load R

###MODIFY SCRIPT TO RUN
python Code/run_cv_cmd_line_script.py -y 2011 -o ar_stdprice_total -r ${resolution} -a ${alg} -h ${hdim}
