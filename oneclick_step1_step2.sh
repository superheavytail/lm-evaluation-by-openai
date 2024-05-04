# This code use 8 GPU by CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 and --nproc_per_node=8
# So if you want to customize GPU utilization, modify that.

# AVAILABLE MODEL TYPE: kullm3, gemma, solar, mistral, openai, koalpaca_v1_1b

SAVE_NAME=ko-gemma-1_1-v2-lstsq_all_second_step240504
MODEL_PATH="/your/model/path"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 step1_generate_answers.py \
--model_name $MODEL_PATH \
--save_name $SAVE_NAME \
--eval_set_path res/user_oriented_instructions_eval.jsonl \
--output_dir ./generated/ \
--model_type gemma \
--debug False

python aggregate_data_shard.py \
--sharded_file_name $SAVE_NAME

python step2_evaluate_by_chatgpt.py \
--generation_result_path "./generated/"$SAVE_NAME".jsonl" \
--evaluation_result_path "./chatgpt_results/"$SAVE_NAME".pkl" \
--use_api