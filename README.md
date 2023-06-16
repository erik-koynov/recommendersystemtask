# Intro
This is a solution to a task, which requires learning a similarity metric
between items from two sets (jobs and job candidates). I implement two
approaches - a baseline, feature-engineering based approach and a deep 
learning based approach. 

## Code
The code is organized as follows:
- scr : contains the runnable scripts
- similarity_learning : a library written for solving the task which
implements all abstractions. It is organized as follows:
    - loading_pipeline : contains abstractions that allow us to easily
  load data into a dataframe format and still preserve the code quality
  and understandability of the objects (i.e. constant property names,
  readable from the source code, instead of only from .columns property
  of a dataframe.)
      - preprocessing_tools : contains all abstractions needed for 
  preprocessing and feature engineering.
    - plotting_tool : mainly used in the EDA
    - inference : contains the code for loading a trained model and predcting
    - feature_engineering

- data : the data used for learning and inference.

## Features:
133 engineered features: among which the most important were the difference
features e.g. between salary expectation and max proposed salary, or the
difference between the seniority min/max required seniority and the job
applicant's seniority.

## Usage
- install via pip install (-e) .
- from similarity_learning import Search
