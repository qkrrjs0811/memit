#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

DATA_DIR='./data/preprocessing/sequential_identical_subjects/each/identical4'
LOG_DIR='./logs/sequential_identical_subjects'

# 이 부분에서 에러 안 나려면, 맨 위에서 'bash'로 실행해야 함
ARGS_LIST=(
	"5 mcf_sequential_identical4_subjects_all"
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
