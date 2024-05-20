# Auto-evaluation for Korean Chat by OpenAI API
모델의 한국어 대화 능력을 ChatGPT가 평가합니다!  
ChatGPT evaluates the model's Korean conversation skills!


## How to evaluate your model
Modify the ```oneclick_step1_step2.sh``` and execute it.  
- It automatically performs step1 and step2, with ChatGPT API usage.  
- It uses about 126K tokens with GPT4-Turbo roughly. (252 API call, with each 500 tokens)
```
sh oneclick_step1_step2.sh
```

## Evaluation Results
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model is omitted because it frequently generates responses in English, even when the input is provided in Korean.
- [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) model sometimes do similarly, therefore, its coherence score is relatively low.

| Type   | Model                                                                        | Fluency (0 - 5) | Coherence (1 - 5) | Accuracy (1 - 5) | Completeness (1 - 5) | Overall Quality (0-5) | 
|--------|------------------------------------------------------------------------------|:---------------:|:-----------------:|:----------------:|:--------------------:|:---------------------:| 
| Closed | GPT-4-Turbo                                                                  |      4.97       |       4.94        |       4.74       |         4.69         |         4.80          | 
| Closed | GPT-3.5-Turbo                                                                |      4.93       |       4.88        |       4.33       |         4.12         |         4.43          |
|        |                                                                              |                 |                   |                  |                      |                       |
| Open   | [KULLM3](https://huggingface.co/nlpai-lab/KULLM3)                            |    **4.83**     |      **4.78**      |     **4.17**     |       **4.23**       |       **4.36**        | 
| Open   | [SOLAR-10.7B-Inst](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) |      4.24       |       3.04        |       3.71       |         3.77         |         3.32          |
| Open   | [KULLM-v2](https://github.com/nlpai-lab/KULLM)                               |      3.70       |       3.25        |       2.82       |         2.48         |         2.80          |
| Open   | [KoAlpaca-1.1b](https://github.com/Beomi/KoAlpaca)                           |      3.90       |       3.25        |       2.60       |         2.37         |         2.67          |

<p align="center">
  <img src="assets/kullm3_instruction_evaluation.png" />
</p>

## How to Reproduce
Edit the contents of the ```*.sh``` files to suit your purpose, and execute!

### step 1
```
sh step1_generate_answers.sh
```
### step 2
```
sh step2_evaluate_by_chatgpt.sh
```
**If you want to use pre-generated model answers and ChatGPT evaluations,**  
you can skip the step 1 and rewrite the ```step2_evaluate_by_chatgpt.sh``` to disable the ```--use_api``` options!

### Supported Models
- nlpai-lab/kullm-v3 (not yet released)
- nlpai-lab/kullm-polyglot-12.8b-v2
- upstage/SOLAR-10.7B-Instruct-v1.0
- mistralai/Mistral-7B-Instruct-v0.2 (excluded since it generates English even though given Korean prompt)
- beomi/KoAlpaca-Polyglot-12.8B
- OpenAI Models (gpt-3.5-turbo, gpt-4-turbo)

## Default Evaluation Template
**Korean evaluation form isn't pretty good, in our experiment result.**  
**So we adopted english form and 'Be Korean language expert' system prompt.**
### System Prompt
```
You're a helpful assistant and a Korean language expert.
```
### User Prompt
```
You will be given evaluation instruction, input and AI-generated response.
Your task is to rate the response on given metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
- Fluency (1-5): The quality of the language used in the translation. A high-quality response should be grammatically correct, idiomatic, and free from spelling and punctuation errors.
- Coherence (1-5): A high score indicates that the response maintains consistent context. A low score is given if the response shifts context or language inappropriately from instruction(e.g. instruction's language is Korean, but response is English).
- Accuracy (1-5) - The correctness of the answer. The answer should be factually correct and directly answer the question asked
- Completeness (1-5) - The extent to which the response covers all aspects of the question. The response should not just address one part of the question, but should provide a comprehensive response.
- Overall Quality (1-5) - The overall effectiveness and excellence of the response, integrating considerations of all above criteria.

Evaluation Steps:
1. Read the instruction and input carefully and understand what it is asking.
2. Read the AI-generated response and Evaluation Criteria.
3. Assign a score for each criterion on a scale of 1 to 5, where 1 is the lowest and 5 is the highest.

Instruction:
{instruction}

Input:
{input}

Response:
{response}

Evaluation Form (scores ONLY):
- Fluency (1-5):
- Coherence (1-5):
- Accuracy (1-5):
- Completeness (1-5):
- Overall Quality (1-5):
```

# Requirements
- torch
- transformers
- [batched-chatgpt](https://github.com/superheavytail/batched-chatgpt)
- fire
- jsonlines
```bash
pip install torch transformers batched-chatgpt fire jsonlines
```
