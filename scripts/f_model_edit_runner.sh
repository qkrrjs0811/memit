#!/bin/bash

# rm -rf ./results

python -u -m falcon.model_edit_runner \
	--alg_name=MEMIT \
	--model_name=gpt2-xl \
	--ds_name=mcf \
	--hparams_fname=gpt2-xl.json \
	--num_edits=2

# python -u -m falcon.model_edit_runner \
# 	--alg_name=ROME \
# 	--model_name=gpt2-xl \
# 	--ds_name=cf \
# 	--hparams_fname=gpt2-xl.json \
# 	--num_edits=1

# python -u -m falcon.model_edit_runner \
# 	--alg_name=ROME \
# 	--model_name=gpt2-xl \
# 	--ds_name=cf \
# 	--hparams_fname=gpt2-xl.json \
# 	--num_edits=1 \
# 	--continue_from_run=run_002

# python -u -m experiments.evaluate \
# 	--alg_name=MEMIT \
# 	--model_name=gpt2-xl \
# 	--ds_name=mcf \
# 	--hparams_fname=gpt2-xl.json \
# 	--num_edits=100

# 근영이 ROME LLaMa3-8b
# python -u -m falcon.model_edit_runner \
# 	--alg_name=ROME \
# 	--model_name=meta-llama/Meta-Llama-3-8B \
# 	--ds_name=cf \
# 	--hparams_fname=meta-llama/Meta-Llama-3-8B.json \
# 	--num_edits=1