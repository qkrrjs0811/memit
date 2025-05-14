1. memit 환경에서 tester.py
2. mergekit 환경에서 merge_runner.py
3. memit 환경에서 tester.py

============================================================================================

# 1. memit 환경에서 독립적으로 편집 및 저장하는 함수 명령어 실행
python -u -m falcon.tester --identical_nums 2 --num_edits_list 125



# 2. mergekit 환경에서 저장한 모델을 병합하는 함수 명령어 실행
### edited된 모델 두 개 병합
python merge_runner.py --model_dirs edited_2_25_1 edited_2_25_2 --merge_methods task_arithmetic ties dare_ties dare_linear della della_linear --lambdas 1.1 1.3 1.5 --densities 0.1 0.3 0.5 0.7 0.9 --epsilons 0.1 0.2 0.3 0.4 --gammas 0.01 0.02 0.03 0.04 0.05 --select_topk 10 20 30 40 50 60 70 80 90

python merge_runner.py --model_dirs edited_2_500_2 gpt2-xl --merge_methods della --lambdas 1.5 --densities 0.7 --epsilons 0.2

python merge_runner.py --model_dirs edited_2_50_1 edited_2_50_2 --merge_methods dare_ties dare_linear della della_linear --lambdas 1.1 1.3 1.5 1.7 --densities 0.1 0.3 0.5 0.7 --epsilons 0.05 0.1 0.2 0.3 0.4


# 3. 기존 방법 및 memit 환경에서 병합한 모델의 편집 성능을 측정하는 함수 명령어 실행
python -u -m falcon.tester --identical_nums 1 --num_edits_list 50




==============================================================================================
# Counterfact data sampling
python sample_json.py --input data/preprocessing/multi_counterfact_20877.json --output data/preprocessing/normal/mcf_sampled_250.json --size 250
