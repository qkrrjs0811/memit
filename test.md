# 다른 데이터셋으로 실험 할 때마다 해야 할 일!!!
- 데이터셋 경로 수정
- 해당 데이터셋에 맞게 zip([4], [5]) 부분도 수정
- edited_models, logs, merged 폴더에 있는 파일 정리!


# 2~3의 과정을 할 때, 아래 사항을 chcek한 이후에 실행하기
- tester.py의 main문에서 merge_test() 함수에 모델명 적기
- 해당 모델명의 hyperparameter에 맞게 merge_conf.yml 파일 수정
- batch 개수에 따라서 merge_conf.yml의 edit_case_1...도 수정
- 해당 내용을 고려하여 test.md의 2~3 명령어 수정

============================================================================================

# 1. memit 환경에서 독립적으로 편집 및 저장하는 함수 명령어 실행
python -m falcon.tester | tee logs/kcc_memit_1_500.log

# 2. mergekit 환경에서 저장한 모델을 병합하는 함수 명령어 실행
python merge_runner.py --output_model_name kcc_merged_2_250_della_30_10

# 3. 기존 방법 및 memit 환경에서 병합한 모델의 편집 성능을 측정하는 함수 명령어 실행
python -u -m falcon.tester | tee logs/kcc_merged_2_250_della_30_10.log



==============================================================================================
# Counterfact data sampling
python sample_json.py --input data/preprocessing/multi_counterfact_20877.json --output data/preprocessing/normal/mcf_sampled_10000.json --size 10000
