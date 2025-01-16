#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

# DATA_DIR='./data/preprocessing/sequential_identical_subjects/each/identical3'
DATA_DIR='./data'
LOG_DIR='./logs'

IN_FILE_PATH=$DATA_DIR"/multi_counterfact.json"
LOG_FILE_PATH=$LOG_DIR"/log_evaluate_identical3_new.txt"
NUM_EDITS=35

sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH
python3 -u -m experiments.summarize --dir_name=MEMIT --runs=run_001

echo "# Terminate all processes!"