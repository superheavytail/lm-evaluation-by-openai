# Enable --use-api option to reproduce with OpenAI API! If not, it use pre-generated result.

python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/upstage-solar.jsonl" \
--evaluation_result_path "./chatgpt_results/upstage-solar.pkl" \
#--use_api
#
python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/kullm_v2.jsonl" \
--evaluation_result_path "./chatgpt_results/kullm_v2.pkl" \
#--use_api
#
python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/gpt_3_5_turbo_0125.jsonl" \
--evaluation_result_path "./chatgpt_results/gpt_3_5_turbo_0125.pkl" \
#--use_api
#
python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/koalpaca_v1_1b.jsonl" \
--evaluation_result_path "./chatgpt_results/koalpaca_v1_1b.pkl" \
#--use_api
#
python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/gpt_4_0125_preview.jsonl" \
--evaluation_result_path "./chatgpt_results/gpt_4_0125_preview.pkl" \
#--use_api

python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/kullm3.jsonl" \
--evaluation_result_path "./chatgpt_results/kullm3.pkl" \
# --use_api

# GEMMA 1.1 2B
python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/gemma_1_1_2b_it.jsonl" \
--evaluation_result_path "./chatgpt_results/gemma_1_1_2b_it.pkl" \
# --use_api

# GEMMA 1.1 7B
python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/gemma_1_1_7b_it.jsonl" \
--evaluation_result_path "./chatgpt_results/gemma_1_1_7b_it.pkl" \
# --use_api