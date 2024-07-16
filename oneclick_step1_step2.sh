# If you want to use pre-generated model answers and ChatGPT evaluations,
# you can remove 'step1(generate)' part
# and modify 'step2(api-evaluation)' part to disable the ```--use_api``` options.

# This code use 8 GPU by CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 and --nproc_per_node=8
# So if you want to customize GPU utilization, modify that.

######################## EXAMPLE 1: KULLM3 ########################
#SAVE_NAME=kullm3
#MODEL_PATH=nlpai-lab/KULLM3
#MODEL_TYPE=model_with_default_chat_template
#EVAL_CATEGORY=chat
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 step1_generate_answers.py \
#--model_name $MODEL_PATH \
#--save_name $SAVE_NAME \
#--eval_category $EVAL_CATEGORY \
#--output_dir ./generated/ \
#--model_type $MODEL_TYPE \
#--debug False &&
#
#python aggregate_data_shard.py \
#--sharded_file_name $SAVE_NAME &&
#
#python step2_evaluate_by_chatgpt.py \
#--generation_result_path "./generated/"$SAVE_NAME".jsonl" \
#--evaluation_result_path "./chatgpt_results/"$SAVE_NAME".pkl" \
#--use_api \
#--eval_category $EVAL_CATEGORY
###################################################################


######################## EXAMPLE 2: gpt-3.5-turbo-0125 ############
#SAVE_NAME=chatgpt_3_5
#MODEL_PATH=gpt-3.5-turbo-0125
#MODEL_TYPE=openai
#EVAL_CATEGORY=chat
#
#python step1_generate_answers.py \
#--model_name $MODEL_PATH \
#--save_name $SAVE_NAME \
#--eval_category $EVAL_CATEGORY \
#--output_dir ./generated/ \
#--model_type $MODEL_TYPE \
#--debug False &&
#
#python step2_evaluate_by_chatgpt.py \
#--generation_result_path "./generated/"$SAVE_NAME".jsonl" \
#--evaluation_result_path "./chatgpt_results/"$SAVE_NAME".pkl" \
#--use_api \
#--eval_category $EVAL_CATEGORY
###################################################################


######################## EXAMPLE 3: gpt-4o-2024-05-13 ############
#SAVE_NAME=gpt_4o_2024_05_13
#MODEL_PATH=gpt-4o-2024-05-13
#MODEL_TYPE=openai
#EVAL_CATEGORY=chat
#
#python step1_generate_answers.py \
#--model_name $MODEL_PATH \
#--save_name $SAVE_NAME \
#--eval_category $EVAL_CATEGORY \
#--output_dir ./generated/ \
#--model_type $MODEL_TYPE \
#--debug False &&
#
#python step2_evaluate_by_chatgpt.py \
#--generation_result_path "./generated/"$SAVE_NAME".jsonl" \
#--evaluation_result_path "./chatgpt_results/"$SAVE_NAME".pkl" \
#--use_api \
#--eval_category $EVAL_CATEGORY
###################################################################


######################## EXAMPLE 4: KULLM-v2 ############
#SAVE_NAME=kullm_v2
#MODEL_PATH=nlpai-lab/kullm-polyglot-12.8b-v2
#MODEL_TYPE=kullm_v2
#EVAL_CATEGORY=chat
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 step1_generate_answers.py \
#--model_name $MODEL_PATH \
#--save_name $SAVE_NAME \
#--eval_category $EVAL_CATEGORY \
#--output_dir ./generated/ \
#--model_type $MODEL_TYPE \
#--debug False &&
#
#python aggregate_data_shard.py \
#--sharded_file_name $SAVE_NAME &&
#
#python step2_evaluate_by_chatgpt.py \
#--generation_result_path "./generated/"$SAVE_NAME".jsonl" \
#--evaluation_result_path "./chatgpt_results/"$SAVE_NAME".pkl" \
#--use_api \
#--eval_category $EVAL_CATEGORY
###################################################################


######################## EXAMPLE 5: SOLAR-10.7B-Inst ############
#SAVE_NAME=SOLAR_10_7B_Inst
#MODEL_PATH=upstage/SOLAR-10.7B-Instruct-v1.0
#MODEL_TYPE=solar
#EVAL_CATEGORY=chat
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 step1_generate_answers.py \
#--model_name $MODEL_PATH \
#--save_name $SAVE_NAME \
#--eval_category $EVAL_CATEGORY \
#--output_dir ./generated/ \
#--model_type $MODEL_TYPE \
#--debug False &&
#
#python aggregate_data_shard.py \
#--sharded_file_name $SAVE_NAME &&
#
#python step2_evaluate_by_chatgpt.py \
#--generation_result_path "./generated/"$SAVE_NAME".jsonl" \
#--evaluation_result_path "./chatgpt_results/"$SAVE_NAME".pkl" \
#--use_api \
#--eval_category $EVAL_CATEGORY
#################################################################


######################## EXAMPLE 6: KoAlpaca-1.1b ###############
#SAVE_NAME=KoAlpaca_1_1b
#MODEL_PATH=beomi/KoAlpaca-Polyglot-12.8B
#MODEL_TYPE=koalpaca_v1_1b
#EVAL_CATEGORY=chat
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 step1_generate_answers.py \
#--model_name $MODEL_PATH \
#--save_name $SAVE_NAME \
#--eval_category $EVAL_CATEGORY \
#--output_dir ./generated/ \
#--model_type $MODEL_TYPE \
#--debug False &&
#
#python aggregate_data_shard.py \
#--sharded_file_name $SAVE_NAME &&
#
#python step2_evaluate_by_chatgpt.py \
#--generation_result_path "./generated/"$SAVE_NAME".jsonl" \
#--evaluation_result_path "./chatgpt_results/"$SAVE_NAME".pkl" \
#--use_api \
#--eval_category $EVAL_CATEGORY
#################################################################
