import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sumeval.metrics.rouge import RougeCalculator
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import logging
from pprint import pprint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_NAME = "yentinglin/TC-Eval"
max_seq_len = 4096

client = OpenAI(timeout=20.0, max_retries=100)
openai_models = {
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-1106",
}

anthropic = Anthropic(timeout=20.0, max_retries=100)
anthropic_models = {
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1.2",
}


# https://github.com/mtkresearch/MR-Models/blob/de1e10d27aed1798a4e1b22d145a45e509652b67/TC-Eval/evaluate.py#L16-L20
def prefix_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() in pred.strip() else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run TC-Eval")
    parser.add_argument(
        "--model",
        type=str,
        default="yentinglin/Taiwan-LLM-7B-v2.0.1-chat",
        help="Model name",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128, help="Max tokens for generation"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tc_eval_results = {}

    is_openai_chat_model = args.model in openai_models
    is_anthropic_chat_model = args.model in anthropic_models
    assert not (
        is_openai_chat_model and is_anthropic_chat_model
    ), "model cannot be both OpenAI and Anthropic chat model"
    if is_openai_chat_model:
        logging.info(f"Using OpenAI chat model: {args.model}")

        def get_openai_chat_response(
            model: str, user_prompt: str, prefill: str = ""
        ) -> str:
            """
            model is "gpt-3.5-turbo-1106" or "gpt-4-1106-preview"
            """
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": prefill.strip()},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            answer = response.choices[0].message.content
            return answer

    elif is_anthropic_chat_model:
        logging.info(f"Using Anthropic chat model: {args.model}")

        def get_anthropic_chat_response(
            model: str, user_prompt: str, prefill: str = ""
        ) -> str:
            while True:
                try:
                    completion = anthropic.completions.create(
                        model=args.model,
                        max_tokens_to_sample=args.max_tokens,
                        temperature=args.temperature,
                        prompt=f"{HUMAN_PROMPT} {user_prompt.strip()}{AI_PROMPT}{prefill.strip()}",
                    )
                except Exception as e:
                    logging.error(f"Error: {e}")
                    continue
                else:
                    break
            return completion.completion

    else:
        logging.info(f"Using LLM model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        sampling_params = SamplingParams(
            temperature=args.temperature, max_tokens=args.max_tokens
        )
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_batched_tokens=40960,
            quantization="AWQ" if "awq" in args.model.lower() else None,
        )

    drcd = load_dataset("yentinglin/TC-Eval", "DRCD", split="test")
    drcd = drcd.map(
        lambda x: {
            "user_prompt": f"請根據以下內容回答問題，且答案需盡可能簡短。注意：答案必須為內容的子字串。\n\n{x['paragraph']}\n\n問題：{x['question']}"
        }
    )
    if is_openai_chat_model:
        answers = []
        for row in tqdm(drcd):
            answer = get_openai_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    elif is_anthropic_chat_model:
        answers = []
        for row in tqdm(drcd):
            answer = get_anthropic_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    else:
        drcd = drcd.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": x["user_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            }
        )
        outputs = llm.generate(drcd["prompt"], sampling_params)
        # sort outputs by request_id
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        # single answer is at outputs[0].outputs[0].text
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]

    scores = [
        max(
            prefix_exact_match(ref, ans) for ref in refs
        )  # prefix exact match, take the max score across all references
        for refs, ans in zip(drcd["references"], answers)
    ]
    drcd_em = sum(scores) / len(scores)
    tc_eval_results["DRCD"] = {"exact_match": drcd_em}
    pprint(tc_eval_results)

    fgc = load_dataset("yentinglin/TC-Eval", "FGC", split="test")
    # 'question' column has artifact in it, remove it
    # 03.維基百科Wikipedia一詞是取自哪兩個字的意義?
    # 04.海倫凱勒出生於哪一個城市？
    # remove artifact r'[0-9]{2}\.'
    fgc = fgc.map(lambda x: {"question": x["question"].replace(r"[0-9]{2}\.", "")})
    fgc = fgc.map(
        lambda x: {
            "user_prompt": f"請根據以下內容回答問題，且答案需盡可能簡短。注意：答案必須為內容的子字串。\n\n{x['paragraph']}\n\n問題：{x['question']}"
        }
    )
    if is_openai_chat_model:
        answers = []
        for row in tqdm(fgc):
            answer = get_openai_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    elif is_anthropic_chat_model:
        answers = []
        for row in tqdm(fgc):
            answer = get_anthropic_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    else:
        fgc = fgc.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": x["user_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            }
        )
        outputs = llm.generate(fgc["prompt"], sampling_params)
        # sort outputs by request_id
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        # single answer is at outputs[0].outputs[0].text
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]
    scores = [
        max(
            prefix_exact_match(ref, ans) for ref in refs
        )  # prefix exact match, take the max score across all references
        for refs, ans in zip(fgc["references"], answers)
    ]
    fgc_em = sum(scores) / len(scores)
    tc_eval_results["FGC"] = {"exact_match": fgc_em}
    pprint(tc_eval_results)

    ttqa = load_dataset(DATASET_NAME, "TTQA", split="test")
    _map_num_to_alph = {i: a for i, a in zip(range(5), "ABCDE")}
    _map_alph_to_num = {a: i for i, a in zip(range(5), "ABCDE")}
    if is_openai_chat_model:
        ttqa = ttqa.map(
            lambda x: {
                "user_prompt": f"問題: {x['question']} \n\n請從以下選項中選擇並回答: {';'.join([f'({_map_num_to_alph[i]}) {tg}' for i, tg in enumerate(x['choices'])])}\n\n只能回答英文字母"
            }
        )
        answers = []
        for row in tqdm(ttqa, desc="TTQA"):
            answer = get_openai_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    elif is_anthropic_chat_model:
        ttqa = ttqa.map(
            lambda x: {
                "user_prompt": f"問題: {x['question']} \n\n請從以下選項中選擇並回答: {';'.join([f'({_map_num_to_alph[i]}) {tg}' for i, tg in enumerate(x['choices'])])}\n\n只能回答英文字母 答案: ("
            }
        )
        answers = []
        for row in tqdm(ttqa, desc="TTQA"):
            answer = get_anthropic_chat_response(
                args.model, row["user_prompt"], prefill="答案: ("
            )
            answers.append(answer)
    else:
        ttqa = ttqa.map(
            lambda x: {
                "user_prompt": f"問題: {x['question']} \n\n請從以下選項中選擇並回答: {';'.join([f'({_map_num_to_alph[i]}) {tg}' for i, tg in enumerate(x['choices'])])}"
            }
        )
        ttqa = ttqa.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": x["user_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + " ("
            }
        )
        outputs = llm.generate(ttqa["prompt"], sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]
    choices = [_map_alph_to_num.get(x[0], "") for x in answers]
    scores = [
        1 if choice == answer else 0 for choice, answer in zip(choices, ttqa["answer"])
    ]
    ttqa_acc = sum(scores) / len(scores)
    tc_eval_results["TTQA"] = {"accuracy": ttqa_acc}
    pprint(tc_eval_results)

    tmmlu = load_dataset(DATASET_NAME, "TMMLU", split="test")
    tmmlu = tmmlu.map(lambda x: {"user_prompt": x["question"]})
    if is_openai_chat_model:
        answers = []
        tmmlu = tmmlu.map(lambda x: {"user_prompt": f"{x['question']}\n\n回答單一英文字母: ("})
        for row in tqdm(tmmlu, desc="TMMLU"):
            answer = get_openai_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    elif is_anthropic_chat_model:
        answers = []
        tmmlu = tmmlu.map(lambda x: {"user_prompt": f"{x['question']}\n\n回答單一英文字母"})
        for row in tqdm(tmmlu, desc="TMMLU"):
            answer = get_anthropic_chat_response(
                args.model, row["user_prompt"], prefill="答案: ("
            )
            answers.append(answer)
    else:
        tmmlu = tmmlu.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": x["user_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + " ("
            }
        )
        outputs = llm.generate(tmmlu["prompt"], sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]
    scores = [
        1 if answer[0] == row["answer"] else 0 for answer, row in zip(answers, tmmlu)
    ]
    tmmlu_acc = sum(scores) / len(scores)
    tc_eval_results["TMMLU"] = {"accuracy": tmmlu_acc}
    pprint(tc_eval_results)

    xsum = load_dataset("yentinglin/TC-Eval", "XSUM_TC", split="test")
    # rename "Unnamed: 0" to "id"
    xsum = xsum.rename_column("Unnamed: 0", "id")
    xsum = xsum.map(lambda x: {"user_prompt": f"{x['document']}\n\n根據上述文章以一句話來總結"})
    if is_openai_chat_model:
        answers = []
        for row in tqdm(xsum, desc="XSUM"):
            answer = get_openai_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    elif is_anthropic_chat_model:
        answers = []
        for row in tqdm(xsum, desc="XSUM"):
            answer = get_anthropic_chat_response(args.model, row["user_prompt"])
            answers.append(answer)
    else:
        xsum = xsum.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": x["user_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            }
        )
        outputs = llm.generate(xsum["prompt"], sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text.strip() for i in range(len(outputs))]
    scorer = RougeCalculator(stemming=True, lang="zh")
    scores = [
        scorer.rouge_2(summary=answer, references=ref)
        for answer, ref in zip(answers, xsum["summary"])
    ]
    xsum_rouge2 = sum(scores) / len(scores)
    tc_eval_results["XSUM"] = {"rouge2": xsum_rouge2}
    pprint(tc_eval_results)

    imdb = load_dataset("yentinglin/TC-Eval", "IMDB_TC", split="test")
    imdb = imdb.map(
        lambda x: {
            "user_prompt": f"評論：{x['text']}\n\n請閱讀以上評論，並回答此評論是正面還是負面，如果是正面，請回答'(1)';，如果是負面，請回答'(0)'"
        }
    )
    if is_openai_chat_model:
        answers = []
        imdb = imdb.map(
            lambda x: {
                "user_prompt": f"評論：{x['text']}\n\n請閱讀以上評論，並回答此評論是正面還是負面，如果是正面，請回答'1';，如果是負面，請回答'0'"
            }
        )
        for row in tqdm(imdb, desc="IMDB"):
            answer = get_openai_chat_response(args.model, row["user_prompt"])
            # use regex to get the 0 or 1. if not found, false
            answer = re.search(r"[0-1]", answer)
            answer = answer.group() if answer else "2"
            answers.append(answer)
            print(answer)
    elif is_anthropic_chat_model:
        answers = []
        for row in tqdm(imdb, desc="IMDB"):
            answer = get_anthropic_chat_response(
                args.model, row["user_prompt"], prefill="答案: ("
            )
            # use regex to get the 0 or 1. if not found, false
            answer = re.search(r"[0-1]", answer)
            answer = answer.group() if answer else "2"
            answers.append(answer)
            print(answer)
    else:
        imdb = imdb.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": x["user_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + " ("
            }
        )
        outputs = llm.generate(imdb["prompt"], sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text.strip() for i in range(len(outputs))]
    scores = [
        1 if answer and answer[0] == str(row["label"]) else 0
        for answer, row in zip(answers, imdb)
    ]
    imdb_acc = sum(scores) / len(scores)
    tc_eval_results["IMDB"] = {"accuracy": imdb_acc}
    pprint(tc_eval_results)

    table = load_dataset("yentinglin/TC-Eval", "PenguinsInTable_TC", split="test")
    # Hacks from https://github.com/mtkresearch/MR-Models/blob/a2d9a3972c4d2a0c982485eb817a1ce4ccdd21a4/TC-Eval/inference/scenarios.py#L168
    table = table.map(lambda row: {"question": row["question"].rstrip("回答：")})
    _map_num_to_alph = {i: a for i, a in zip(range(5), "ABCDE")}
    table = table.map(
        lambda row: {
            "user_prompt": f"{row['question']} \n請從以下選項中選擇並回答: {';'.join([f'({_map_num_to_alph[i]}) {tg}' for i, tg in enumerate(row['choices'])])}"
        }
    )
    if is_openai_chat_model:
        choices = []
        for row in tqdm(table, desc="PenguinsInTable"):
            answer = get_openai_chat_response(
                args.model, row["user_prompt"] + "\n只能回答單一英文字母"
            )
            # use regex to get the first English alphabet, if not found, use empty string
            answer = re.search(r"[A-E]", answer)
            answer = answer.group() if answer else ""
            # map
            answer = _map_alph_to_num.get(answer, "")
            choices.append(answer)
            print(answer)
    elif is_anthropic_chat_model:
        choices = []
        for row in tqdm(table, desc="PenguinsInTable"):
            answer = get_anthropic_chat_response(
                args.model, row["user_prompt"] + "\n只能回答單一英文字母", prefill="答案: ("
            )
            # use regex to get the first English alphabet, if not found, use empty string
            answer = re.search(r"[A-E]", answer)
            answer = answer.group() if answer else ""
            # map
            answer = _map_alph_to_num.get(answer, "")
            choices.append(answer)
            print(answer)
    else:
        table = table.map(
            lambda row: {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["user_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + " ("
            }
        )
        outputs = llm.generate(table["prompt"], sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]
        choices = [_map_alph_to_num.get(x[0], "") for x in answers]
    scores = [
        1 if choice == answer else 0 for choice, answer in zip(choices, table["answer"])
    ]
    table_acc = sum(scores) / len(scores)
    tc_eval_results["PenguinsInTable"] = {"accuracy": table_acc}
    pprint(tc_eval_results)


if __name__ == "__main__":
    main()
