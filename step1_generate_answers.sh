# kullm v2
#python step1_generate_answers.py \
#--model_name nlpai-lab/kullm-polyglot-12.8b-v2 \
#--save_name kullm_v2 \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type kullm \
#--debug False

# KULLM3
# python step1_generate_answers.py \
# --model_name nlpai-lab/KULLM3 \
# --save_name kullm3 \
# --eval_set_path res/user_oriented_instructions_eval.jsonl \
# --output_dir ./generated/ \
# --model_type kullm3 \
# --debug False

# solar instruct
#python step1_generate_answers.py \
#--model_name upstage/SOLAR-10.7B-Instruct-v1.0 \
#--save_name upstage-solar \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type solar \
#--debug False

# mistral 7B (excluded since it often say English, though Korean context.)
#python step1_generate_answers.py \
#--model_name mistralai/Mistral-7B-Instruct-v0.2 \
#--save_name mistral \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_d./generated/ \
#--model_type mistral \
#--debug False

# beomi/koalpaca_v1.1b
#python step1_generate_answers.py \
#--model_name beomi/KoAlpaca-Polyglot-12.8B \
#--save_name koalpaca_v1_1b \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type koalpaca_v1_1b \
#--debug False


# gpt-3.5 turbo
#python step1_generate_answers.py \
#--model_name gpt-3.5-turbo-0125 \
#--save_name gpt_3_5_turbo_0125 \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type openai \
#--debug False

# gpt-4 turbo
#python step1_generate_answers.py \
#--model_name gpt-4-0125-preview \
#--save_name gpt_4_0125_preview \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type openai \
#--debug False

# kullm3 (templated) (local)
#python step1_generate_answers.py \
#--model_name nlpai-lab/KULLM3 \
#--save_name kullm3 \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type kullm3 \
#--debug False

# gemma_1.1_2b
#python step1_generate_answers.py \
#--model_name google/gemma-1.1-2b-it \
#--save_name gemma_1_1_2b_it \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type gemma \
#--debug False

# gemma_1.1_7b
#python step1_generate_answers.py \
#--model_name google/gemma-1.1-7b-it \
#--save_name gemma_1_1_7b_it \
#--eval_set_path res/user_oriented_instructions_eval.jsonl \
#--output_dir ./generated/ \
#--model_type gemma \
#--debug False