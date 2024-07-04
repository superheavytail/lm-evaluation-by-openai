from pathlib import Path
from collections import defaultdict
from itertools import chain
import fire
import jsonlines
import pickle
import re
import json
import time
import numpy as np

from batched_chatgpt import call_chatgpt

GREEN = '\033[92m'
CYAN = '\033[96m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'

CHATGPT_MODEL_NAME = "gpt-4o-2024-05-13"


# test function
def pattern_check(p1, p2, p3, s):
    print(s)
    if len(p1.findall(s)) == 5:
        print("☆☆☆p1 matched")
        print(p1.findall(s))
    elif len(p2.findall(s)) == 5:
        print("☆☆☆p2 matched")
        print(p2.findall(s))
    elif len(p3.findall(s)) == 5:
        print("☆☆☆p3 matched")
        print(p3.findall(s))
    else:
        raise AssertionError


def parse_chatgpt_answer(s: str):
    # score or N/A are possible

    # pattern 1: ideal situation, "- Fluency (1-5): 0"
    p1 = re.compile(
        r".*((?:fluency)|(?:coherence)|(?:accuracy)|(?:completeness)|(?:overall quality))\s*(?:\(?.*\))?\s*:\s*(\d|N/A)",
        re.IGNORECASE
    )
    # pattern 2: "- Fluency (5)"
    p2 = re.compile(
        r".*((?:fluency)|(?:coherence)|(?:accuracy)|(?:completeness)|(?:overall quality))\s*\((\d|N/A)\)",
        re.IGNORECASE
    )
    # pattern 3: "- Fluency (1-5)" or "- Fluency (1/5)"
    p3 = re.compile(
        r".*((?:fluency)|(?:coherence)|(?:accuracy)|(?:completeness)|(?:overall quality))\s*\((\d|N/A)[-/]5\)",
        re.IGNORECASE
    )

    scores = {}
    for p in (p1, p2, p3):
        l = p.findall(s)  # [('Fluency', '5'), ('Coherence', '1'), ('Accuracy', '2'), ('Completeness', '2'), ('Overall Quality', '2')]
        if len(l) != 5:
            continue
        else:
            for item in l:
                if item[1] == 'N/A':
                    score = 1  # score is 1 when 'N/A' given.
                else:
                    score = int(item[1])
                scores[item[0].lower()] = score
            break
    assert set(scores.keys()) == {'fluency', 'coherence', 'accuracy', 'completeness', 'overall quality'}, f"FAIL:\n{s}"
    return scores


def chat_evaluate(
        generation_result_path: str,
        evaluation_result_path: str,
        use_api: bool = False,
):
    # load generation result
    j = jsonlines.open(generation_result_path)
    gen_results = [e for e in j.iter()]
    for key in ['instruction', 'input', 'answer']:
        assert key in gen_results[0].keys()

    # load AI evaluation template
    with open("evaluation_templates/en_template_for_llm_eval_input.txt", 'rt') as f:
        template_with_input = f.read()
    with open("evaluation_templates/en_template_for_llm_eval_noinput.txt", 'rt') as f:
        template_with_noinput = f.read()

    # Apply template
    chatgpt_inputs = []
    for gen in gen_results:
        if gen['input']:
            chatgpt_inputs.append(template_with_input.format_map({
                'instruction': gen['instruction'],
                'input': gen['input'],
                'response': gen['answer']
            }))
        else:
            chatgpt_inputs.append(template_with_noinput.format_map({
                'instruction': gen['instruction'],
                'response': gen['answer']
            }))

    # Do AI evaluation and save result
    if use_api:
        print(f"Current ChatGPT model: {CHATGPT_MODEL_NAME}")
        evaluation_results = call_chatgpt(
            chatgpt_inputs,
            system_message="You're a helpful assistant and a Korean language expert.",
            model_name=CHATGPT_MODEL_NAME,
            pkl_path=evaluation_result_path,
            chunk_size=20
        )
    else:
        with open(evaluation_result_path, 'rb') as f:
            evaluation_results = pickle.load(f)

    # Parse AI response
    not_parseable_count = 0
    scores = {
        "fluency": [],
        "coherence": [],
        "accuracy": [],
        "completeness": [],
        "overall quality": [],
    }
    for result in evaluation_results:
        try:
            parsed = parse_chatgpt_answer(result)
            for k, v in parsed.items():
                scores[k].append(v)
        except AssertionError:
            not_parseable_count += 1
    assert len(scores['fluency']) + not_parseable_count == len(evaluation_results)

    scores = dict({'fluency ': scores['fluency']}, **scores)  # For beautiful display
    del (scores['fluency'])

    # Print parsed result
    # scores and abandoned number will be printed
    averages = {k: np.average(v) for k, v in scores.items()}
    stddev = {k: np.std(v) for k, v in scores.items()}
    print()
    print(f"RESULT PATH: {evaluation_result_path}")
    print(BLUE + "===================================" + END)
    print(CYAN + f"  The number of all examples: " + END + f"{len(evaluation_results)}")
    print(GREEN + f"  Parsing successes: " + END + f"{len(evaluation_results) - not_parseable_count}")
    print(RED + f"  Parsing failed: " + END + f"{not_parseable_count}")
    print(BLUE + "===================================" + END)
    print(BOLD + "   CRITERIA\tAVG\tSTD" + END)
    print("-----------------------------------")
    for key in scores:
        print(f"{key}\t{round(averages[key], 2)}\t{round(stddev[key], 2)}")
        print("-----------------------------------")
    print()


def hallucination_evaluate(
        generation_result_path: str,
        evaluation_result_path: str,
        eval_set_path: str,
        use_api: bool = False,
        num_items_in_one_api_call: int = 8,
):
    # load generation result
    j = jsonlines.open(generation_result_path)
    gen_results = [e for e in j.iter()]
    for key in ['prompt', 'completion']:
        assert key in gen_results[0].keys()

    # load 'context' data from evaluation data
    with open(eval_set_path, 'rt', encoding='utf-8') as f:
        eval_set = json.load(f)
    contexts = [e['context'] for e in chain(*eval_set.values())]

    # load AI evaluation template
    with open("evaluation_templates/en_template_for_halluci_eval.txt", 'rt') as f:
        halluci_template = f.read()

    # Make string chunk, since we're going to evaluate multiple instances in one API call.
    item_chunks = []
    tmp_item_chunk = []
    assert len(gen_results) == len(contexts)
    for i, (gen_result, context) in enumerate(zip(gen_results, contexts)):
        question = gen_result['prompt']
        response = gen_result['completion']
        remainder = i % num_items_in_one_api_call

        item = (f"<Item {remainder + 1}>\n"
                f"[Reference]\n"
                f"{context}\n"
                f"[Question]\n"
                f"{question}\n"
                f"[Response]\n"
                f"{response}\n"
                f"<Item {remainder + 1}> End\n")
        result_sheet = (f"[Item {remainder + 1}]\n"
                        f"is_hallucinated: \n"
                        f"is_helpful: \n")
        tmp_item_chunk.append((item, result_sheet))

        if i % num_items_in_one_api_call == num_items_in_one_api_call - 1 or i == len(gen_results) - 1:
            item_chunks.append((
                "\n".join([e[0] for e in tmp_item_chunk]),
                "".join([e[1] for e in tmp_item_chunk]),
            ))
            tmp_item_chunk = []

    # Apply template
    item_chunks = [halluci_template.format_map({
        "items_ref_ques_resp": e[0],
        "result_sheets": e[1],
    }) for e in item_chunks]

    # Do AI evaluation and save result
    if use_api:
        print("Initiating ChatGPT API...")
        print(f"Current ChatGPT model: {CHATGPT_MODEL_NAME}")
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)

        evaluations = call_chatgpt(
            item_chunks,
            system_message="You're an helpful assistant.",
            model_name=CHATGPT_MODEL_NAME,
            pkl_path=evaluation_result_path,
            chunk_size=5,
            sleep_between_chunk=2,
            temperature=0
        )
    else:
        with open(evaluation_result_path, 'rb') as f:
            evaluations = pickle.load(f)['completions']

    # Parse AI response
    parsed_data = []
    p1 = r'\[Item\s*(\d+)\]\n+is_hallucinated:\s*(\w+)\n+is_helpful:\s*(\w+)'
    pattern = re.compile(p1)

    for evaluation in evaluations:
        matches = pattern.findall(evaluation)
        parsed_data.extend(
            [{'Item': int(item), 'is_hallucinated': hallucinated, 'is_helpful': helpful} for
             item, hallucinated, helpful in matches]
        )

    if len(parsed_data) == len(contexts):
        # Collapse by category of hallucination
        items_by_category = dict.fromkeys(eval_set)
        cnt = 0
        for key in eval_set:
            items_by_category[key] = parsed_data[cnt:cnt + len(eval_set[key])]
            cnt += len(eval_set[key])

        # Count and score
        scores = {}
        total_examples = 0
        total_hallucinated_count = 0
        total_helpful_count = 0
        for key, value in items_by_category.items():
            hallucinated_count = len([1 for e in value if e['is_hallucinated'] == 'yes'])
            helpful_count = len([1 for e in value if e['is_helpful'] == 'yes'])
            scores[key] = {
                "num_examples": len(value),
                "hallucinated_count": hallucinated_count,
                "hallucinated_rate": hallucinated_count / len(value),
                "helpful_count": helpful_count,
                "helpful_rate": helpful_count / len(value),
            }
            total_examples += len(value)
            total_hallucinated_count += hallucinated_count
            total_helpful_count += helpful_count
        scores['Global Score (Micro averaging)'] = {
            'num_examples': total_examples,
            'hallucinated_count': total_hallucinated_count,
            'hallucinated_rate': total_hallucinated_count / total_examples,
            'helpful_count': total_helpful_count,
            'helpful_rate': total_helpful_count / total_examples,
        }

        # Display the scores
        print(BLUE + "===================================" + END)
        print(CYAN + f"  The number of all examples: " + END + f"{len(contexts)}")
        print(BLUE + "===================================" + END)
        for key, value in scores.items():
            print("-------------------------------")
            print(RED + f"{key}" + END)
            print("-------------------------------")
            for k, v in value.items():
                if k.startswith("halluci"):
                    if k.endswith("rate"):
                        print(GREEN + f"{k}\t" + END + f"{round(v, 3)}")
                    else:
                        print(CYAN + f"{k}\t" + END + f"{round(v, 3)}")
                elif k.startswith("helpful"):
                    if k.endswith("rate"):
                        print(GREEN + f"{k}\t\t" + END + f"{round(v, 3)}")
                    else:
                        print(CYAN + f"{k}\t\t" + END + f"{round(v, 3)}")
                elif k.startswith("num_examples"):
                    print(CYAN + f"{k}\t\t" + END + f"{round(v, 3)}")
                else:
                    raise ValueError
    else:
        print("Cannot parse all ChatGPT's answers correctly!")
        raise ValueError


def main(
    category: str,
    generation_result_path: str,
    evaluation_result_path: str,
    eval_set_path: str = None,
    use_api: bool = False,
    num_items_in_one_api_call: int = 8,
):
    if category == "chat":
        chat_evaluate(generation_result_path, evaluation_result_path, use_api)
    elif category == 'hallucination':
        hallucination_evaluate(
            generation_result_path,
            evaluation_result_path,
            eval_set_path, use_api,
            num_items_in_one_api_call,
        )
    else:
        raise ValueError


if __name__ == '__main__':
    fire.Fire(main)
