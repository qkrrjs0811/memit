#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

DATA_DIR='./data/preprocessing/multiple_identical_subjects'
LOG_DIR='./logs/multiple_identical_subjects'

# 이 부분에서 에러 안 나려면, 맨 위에서 'bash'로 실행해야 함
ARGS_LIST=(
	"1000 mcf_multiple_identical_subjects_1000_10:0"
	"1000 mcf_multiple_identical_subjects_1000_0:10"
	"1000 mcf_multiple_identical_subjects_1000_9:1"
	"1000 mcf_multiple_identical_subjects_1000_8:2"
	"1000 mcf_multiple_identical_subjects_1000_7:3"
	"1000 mcf_multiple_identical_subjects_1000_6:4"
	"1000 mcf_multiple_identical_subjects_1000_5:5"
	"1000 mcf_multiple_identical_subjects_1000_4:6"
	"1000 mcf_multiple_identical_subjects_1000_3:7"
	"1000 mcf_multiple_identical_subjects_1000_2:8"
	"1000 mcf_multiple_identical_subjects_1000_1:9"
)


for ARGS_ in "${ARGS_LIST[@]}"; do
	read -r -a ARGS <<< "$ARGS_"

	NUM_EDITS=${ARGS[0]}
	FILE_PATH=${ARGS[1]}

	IN_FILE_PATH=$DATA_DIR"/"$FILE_PATH".json"
	LOG_FILE_PATH=$LOG_DIR"/log_"$FILE_PATH"_batch"$NUM_EDITS".txt"

	echo "num_edits : $NUM_EDITS"
	echo "file_path : $FILE_PATH"
	echo "in_file_path : $IN_FILE_PATH"
	echo "log_file_path : $LOG_FILE_PATH"
	echo ""

	sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH
done

echo "# Terminate all processes!"
