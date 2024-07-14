from pathlib import Path
from itertools import chain
import os
import fire
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
from batched_chatgpt import call_chatgpt
import jsonlines

from step2_evaluate_by_chatgpt import GREEN, RED, END


eval_category_path = {
    "chat": "res/kor_user_oriented_instructions_eval.jsonl",
    "hallucination": "res/halluci_crafted.json"
}


def print_(x, local_rank=None):
    if local_rank == 0:
        print(x)


def main(
    model_name,
    save_name,
    eval_category,
    output_dir,
    model_type,
    debug: bool = False
):
    # Make save_path and check if it is safe
    if debug:
        save_name = save_name + "_debug"
    save_path = Path(output_dir) / save_name
    save_path = save_path.with_suffix(".jsonl")
    if save_path.exists():
        raise FileExistsError(f"{save_path} exists!")

    print(GREEN + "=== model name ===" + END)
    print(model_name)
    print(GREEN + "==================" + END)
    print(os.getcwd())

    # prepare evaluation set
    # It can be refactored to have more readability, but move fast.
    eval_set_path = eval_category_path[eval_category]
    if eval_category == 'chat':
        eval_set = [e['prompt'] for e in jsonlines.open(eval_set_path).iter()]
    elif eval_category == 'hallucination':
        with open(eval_set_path, 'rt', encoding='utf-8') as f:
            eval_set = json.load(f)
        eval_set = [e['question'] for e in chain(*eval_set.values())]
    else:
        raise ValueError
    eval_set = eval_set[:3] if debug else eval_set

    generated_answers = []
    device = None

    if model_type == 'openai':
        generated_answers = call_chatgpt(eval_set, chunk_size=20, model_name=model_name, temperature=0)
    else:
        # for multi-GPU inference
        try:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            eval_set = eval_set[local_rank::world_size]
            save_path = (save_path.parent / (save_path.stem + f'-rank{local_rank}')).with_suffix('.jsonl')
            device = f"cuda:{local_rank}"
        except KeyError:
            device = "cuda"
            local_rank = None  # for print_ function

        # This models need special generation strategy
        if model_type in ['solar', 'mistral', 'kullm_v2', 'koalpaca_v1_1b', 'hyundai_llm']:
            # TODO currently this 'if' block cannot deal with 'hallucination' category
            # There exists more simple way, but let us move fast.
            # model, tokenizer load
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype='auto',
                    attn_implementation='sdpa').to(device)
            except ValueError:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype='auto').to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # building input text
            # solar uses apply_chat_template but kullm doesn't
            if model_type == 'solar' or model_type == 'mistral':
                input_template_texts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True) for t in eval_set]

            elif model_type in ['kullm_v2', 'koalpaca_v1_1b', 'hyundai_llm']:
                # for generate
                model.config.pad_token_id = model.config.eos_token_id

                with open(f"lm_templates/{model_type}.json", 'rt') as f:
                    template = json.load(f)

                input_template_texts = []
                for e in eval_set:
                    input_text = template['prompt_no_input'].format_map({
                        'instruction': e,
                    })
                    input_template_texts.append(input_text)
            else:
                raise NotImplementedError

            input_ids = [tokenizer(e, return_tensors='pt')["input_ids"].to(model.device) for e in input_template_texts]

            # do generation
            if not tokenizer.pad_token_id and tokenizer.eos_token_id:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            for item in tqdm(input_ids):
                res = model.generate(
                    item,
                    use_cache=True,
                    max_new_tokens=min(2048-len(item[0]), 512),  # it can be refactored, but low-performance model can repeat useless word. so it seems good to have shorter max_new_tokens.
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                )
                try:
                    model_answer = tokenizer.decode(res[0][len(item[0]):], skip_special_tokens=True)
                except IndexError:
                    print("model generate nothing!! Full model answer:")
                    model_answer = "(no answer from model)"  # Some models rarely generate nothing.

                generated_answers.append(model_answer)
                if debug:
                    print(model_answer)
        else:
            # Prepare model, tokenizer, HF pipeline
            print_(f"model_type: {model_type}", local_rank=local_rank)
            print_(RED + "Using model's default chat template..." + END, local_rank=local_rank)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", attn_implementation='sdpa')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False,
                            do_sample=False, use_cache=True)

            # Prepare data
            eval_set_prepared = [
                [{'role': 'user', 'content': e}]
                for e in eval_set
            ]

            for e in tqdm(eval_set_prepared):
                inputs = tokenizer.apply_chat_template(e, tokenize=False, add_generation_prompt=True)
                generated_answers.append(pipe(inputs, max_new_tokens=2048)[0]['generated_text'])
                print(generated_answers[-1])

    # saving results
    # zip instruction, input, answer
    results = []

    for prompt, completion in zip(eval_set, generated_answers):
        results.append({
            'prompt': prompt,
            'completion': completion
        })

    print(GREEN + "saving result..." + END) if device == 'cuda' or not device else ...
    with open(save_path, "wt", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)  # set ensure_ascii False for Korean language
            f.write("\n")  # since it's jsonl

    if debug:
        for result in results:
            print(f"{result['prompt']=}")
            print(f"{result['completion']=}")


if __name__ == '__main__':
    fire.Fire(main)
