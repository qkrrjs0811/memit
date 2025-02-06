#!/bin/bash
set -e

python3 -u -m experiments.evaluate \
    --alg_name=MEMIT \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --num_edits=35

# python3 -u -m experiments.evaluate \
#     --alg_name=MEMIT \
#     --model_name=EleutherAI/gpt-j-6B \
#     --hparams_fname=EleutherAI_gpt-j-6B.json \
#     --num_edits=10000 \
#     --use_cache