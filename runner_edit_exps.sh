#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

DATA_DIR='./data/preprocessing'
LOG_DIR='./logs/sr_swap'

# 이 부분에서 에러 안 나려면 'bash'로 실행해야 함
ARGS_LIST=(
	"1000 identical1_ext_rn_1000_sr_swap"
	"1000 identical2_ext_n_1000_sr_swap"
	"105 identical3_all_105_sr_swap"
	"20 identical4_all_20_sr_swap"
	"1000 identical1_ext_rn_1000_sr_swap_post"
	"1000 identical2_ext_n_1000_sr_swap_post"
	"105 identical3_all_105_sr_swap_post"
	"20 identical4_all_20_sr_swap_post"
)


for ARGS_ in "${ARGS_LIST[@]}"; do
	read -r -a ARGS <<< "$ARGS_"

	NUM_EDITS=${ARGS[0]}
	FILE_PATH=${ARGS[1]}

	IN_FILE_PATH=$DATA_DIR"/multi_counterfact_"$FILE_PATH".json"
	LOG_FILE_PATH=$LOG_DIR"/log_"$FILE_PATH"_batch"$NUM_EDITS".txt"

	echo "num_edits : $NUM_EDITS"
	echo "file_path : $FILE_PATH"
	echo "in_file_path : $IN_FILE_PATH"
	echo "log_file_path : $LOG_FILE_PATH"
	echo ""

	sh ./scripts/f_model_edit_runner.sh $NUM_EDITS $IN_FILE_PATH > $LOG_FILE_PATH
done

echo "# Terminate all processes!"