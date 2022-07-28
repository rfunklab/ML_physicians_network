# Machine Learning Pipeline for TDA of Physicians' Networks

Compute persistence images for the pysicians' network data and then use K-fold cross validation (CV) to select the best algorithm and parameters (for the algorithm and the persistence image weight function).

## Current Pipeline
1. Generate persistence diagrams (PDs) from the graph data (not included here) from a file of HSA IDs.
    - [Code/generate_PDs_as_strings.py][1]
2. Generate Test and Fold indicies for the PDs.
    - [Code/generate_kfolds_indices.py][2]
3. Run cross validation.
   - Module: paths, parameter values, functions, and other useful information
      - [Code/modules/cv_prep_vars.py][3]
      - Most of the editing and generic (algorithm agnostic) information is here
    - Command line script: Runs CV for a given year, outcome, pixel resolution, H dimension, scoring metric, and k/test percent information
      - [Code/run_cv_cmd_line_script.py][4]
    - This script should be run through a job scheduler because it can take days for some algorithms. See the "submit_scritps" directory, jobs_submit.sh (to easily submit and name jobs), and eg_submit_cmd example file.

## Future Directions
1. Edit graph script:
   - [ ] Option to run before generating PDs
2. Generate PDs:
   - [ ] Fixes to this script, including making it a command line script
3. Generate Test and Fold indicies
4. Generate CV Data:
   - [X] Make more reproducible and faster
   - [ ] Finish testing modifications and replace original version
5. Select Best Parameters/Model(s)
   - [ ] Determine how to select the best models
6. Fit and Explore Best Model/s
   - [ ] Fit best models and explore the results
7. Sensitivity Tests
   - [ ] Re-do steps 4-6 with different outcome definitions and different pixel resolutions

[1]: https://github.com/rfunklab/ML_physicians_network/blob/master/Code/generate_PDs_as_strings.py
[2]: https://github.com/rfunklab/ML_physicians_network/blob/master/Code/generate_kfolds_indices.py
[3]: https://github.com/rfunklab/ML_physicians_network/blob/master/Code/modules/cv_prep_vars.py
[4]: https://github.com/rfunklab/ML_physicians_network/blob/master/Code/run_cv_cmd_line_script.py
