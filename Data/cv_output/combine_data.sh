#!/bin/bash

outcome=${1}
outcome_type=${2}
seed=${3}
resolution=${4}

# Initialize files
dim0_file=outcomes/${outcome_type}/${outcome}/cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-${resolution}_hdim-0_metric-mse_seed-${seed}_desc-cv_results.txt
dim1_file=outcomes/${outcome_type}/${outcome}/cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-${resolution}_hdim-1_metric-mse_seed-${seed}_desc-cv_results.txt

cp ./header.txt ${dim0_file}
cp ${dim0_file} ${dim1_file}

# Combine data
tail -n +2 cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-${resolution}_hdim-0_metric-mse_seed-${seed}_v-* | grep -v ">" >> ${dim0_file}
tail -n +2 cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-${resolution}_hdim-1_metric-mse_seed-${seed}_v-* | grep -v ">" >> ${dim1_file}

# Delete old files
rm -f cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-${resolution}_hdim-*_v-*