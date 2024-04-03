from pathlib import Path
from collections import defaultdict
import fire
import jsonlines
import pickle
import re
import numpy as np

from batched_chatgpt import call_chatgpt

GREEN = '\033[92m'
CYAN = '\033[96m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'

CHATGPT_MODEL_NAME = "gpt-4-0125-preview"


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


def main(
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
    del(scores['fluency'])

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


if __name__ == '__main__':
    fire.Fire(main)
