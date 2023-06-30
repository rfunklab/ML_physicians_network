#!/bin/bash

outcome=${1}
outcome_type=${2}

# Initialize files
dim0_file=outcome/${outcome_type}/${outcome}/cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-mid_hdim-0_metric-mse_seed-3667424171_desc-cv_results.txt
dim1_file=outcome/${outcome_type}/${outcome}/cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-mid_hdim-1_metric-mse_seed-3667424171_desc-cv_results.txt

cp ./header.txt ${dim0_file}
cp ${dim0_file} ${dim1_file}

# Combine data
tail -n +2 outcome/${outcome_type}/${outcome}/cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-mid_hdim-0_metric-mse_seed-3667424171_v-* | grep -v ">" >> ${dim0_file}
tail -n +2 outcome/${outcome_type}/${outcome}/cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-mid_hdim-1_metric-mse_seed-3667424171_v-* | grep -v ">" >> ${dim1_file}

# Delete old files
rm -f outcome/${outcome_type}/${outcome}/cv_alg-en_outcome-${outcome}_year-2011_region-great_lakes_res-mid_hdim-*_v-*