#!/bin/sh

if [ 0 -eq $# ];
then
	echo "command : sh runner.sh (task_option) (log_file_name)"
	echo "task_option is [1] : scripts/f_model_edit_runner.sh"
	echo "task_option is [2] : scripts/f_causal_trace.sh"
	echo "task_option is [3] : scripts/f_summarize.sh"
else
	if [ 1 -eq $1 ];
	then
		sh ./scripts/f_model_edit_runner.sh > logs/$2

	elif [ 2 -eq $1 ];
	then
		sh ./scripts/f_causal_trace.sh > logs/$2

	elif [ 3 -eq $1 ];
	then
		sh ./scripts/f_summarize.sh > logs/$2
	fi
fi