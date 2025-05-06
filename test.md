1. memit 환경에서 tester.py
2. mergekit 환경에서 merge_runner.py
3. memit 환경에서 tester.py

============================================================================================

# 1. memit 환경에서 독립적으로 편집 및 저장하는 함수 명령어 실행
python -u -m falcon.tester --identical_nums 2 2 2 2 2 2 2 --num_edits_list 10 25 50 500 1500 2500 5000



# 2. mergekit 환경에서 저장한 모델을 병합하는 함수 명령어 실행
### edited된 모델 두 개 병합
python merge_runner.py --model_dirs edited_2_25_1 edited_2_25_2 --merge_methods task_arithmetic ties dare_ties dare_linear breadcrumbs breadcrumbs_ties model_stock della della_linear sce --densities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --epsilons 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

### pruning만한 모델 생성 (edited + origin)
python merge_runner.py --model_dirs edited_2_10_1 gpt2-xl --merge_methods task_arithmetic ties dare_ties dare_linear breadcrumbs breadcrumbs_ties model_stock della della_linear sce --densities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --epsilons 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

python merge_runner.py --model_dirs edited_2_10_2 gpt2-xl --merge_methods task_arithmetic ties dare_ties dare_linear breadcrumbs breadcrumbs_ties model_stock della della_linear sce --densities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --epsilons 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9



python merge_runner.py --model_dirs edited_2_10_1 edited_2_10_2 --merge_methods task_arithmetic della --densities 0.1 0.2 --epsilons 0.1 0.2


# 3. 기존 방법 및 memit 환경에서 병합한 모델의 편집 성능을 측정하는 함수 명령어 실행
python -u -m falcon.tester --identical_nums 1 --num_edits_list 20




==============================================================================================
# Counterfact data sampling
python sample_json.py --input data/preprocessing/multi_counterfact_20877.json --output data/preprocessing/normal/mcf_sampled_10000.json --size 10000
