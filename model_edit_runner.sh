rm -rf ./results

python -u -m falcon.model_edit_runner \
	--alg_name=MEMIT \
	--model_name=gpt2-xl \
	--ds_name=mcf \
	--hparams_fname=gpt2-xl.json \
	--num_edits=100

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