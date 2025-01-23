#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

# DATA_DIR='./data/preprocessing/sequential_identical_subjects/each/identical3'
# LOG_DIR='./logs/evaluate_matrix/multiple'
# IN_FILE_PATH=$DATA_DIR"/mcf_sequential_identical3_subjects_all.json"
# LOG_FILE_PATH=$LOG_DIR"/log_mcf_identical3_all_35*3.txt"
# NUM_EDITS=35
# sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH


# Multiple
DATA_DIR='./data/preprocessing'
LOG_DIR='./logs/evaluate_matrix/multiple'

IN_FILE_PATH=$DATA_DIR"/multi_counterfact_identical1_ext_rn_1000.json"
LOG_FILE_PATH=$LOG_DIR"/log_mcf_identical1_ext_rn_1000*1.txt"
NUM_EDITS=1000
sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH

IN_FILE_PATH=$DATA_DIR"/multi_counterfact_identical2_ext_n_1000.json"
LOG_FILE_PATH=$LOG_DIR"/log_mcf_identical2_ext_n_1000*1.txt"
NUM_EDITS=1000
sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH

IN_FILE_PATH=$DATA_DIR"/multi_counterfact_identical3_all_105.json"
LOG_FILE_PATH=$LOG_DIR"/log_mcf_identical3_all_105*1.txt"
NUM_EDITS=105
sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH

IN_FILE_PATH=$DATA_DIR"/multi_counterfact_identical4_all_20.json"
LOG_FILE_PATH=$LOG_DIR"/log_mcf_identical4_all_20*1.txt"
NUM_EDITS=20
sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH


python3 -u -m experiments.summarize --dir_name=MEMIT --runs=run_000,run_001,run_002,run_003

echo "# Terminate all processes!"