import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from langchain.chat_models import ChatLiteLLM
from langchain.llms import Anthropic

logging.basicConfig(level=logging.INFO)

NEED_REF_CATS = ["math", "reasoning", "coding"]

pairwise_pk_eval_template = """[用戶問題]
{question}

[助理A的答案開始]
{answer_a}
[助理A的答案結束]

[助理B的答案開始]
{answer_b}
[助理B的答案結束]

請充分比較兩個AI助理對用戶問題的回答，並做出公正的判斷，以評估回答的質量。你應該選擇更符合用戶指示並更好地回答用戶問題的助理。你的評估應該考慮因素，例如回答的有用性、相關性、準確性、深度、創造力和細節水平。注意，簡體中文不能使用，如果助理使用簡體中文要給最低分；文字必須符合台灣繁體中文。英文只接受專有名詞。通過比較兩個回答並提供簡短的解釋來開始你的評估。避免任何立場偏見，並確保回答的呈現順序不會影響你的決定。不要因回答的長度影響你的評估。不要偏愛某個助理的名稱。請盡量客觀。在提供解釋後，通過嚴格遵循以下格式輸出你的最終判決："[[A]]"如果助理A更好，"[[B]]"如果助理B更好，和"[[C]]"打成平手。"""

pairwise_pk_reference_eval_template = """[用戶問題]
{question}

[英文參考答案開始，請只考慮答案和推理過程，忽略語言]
{ref_answer_1}
[英文參考答案結束，請只考慮答案和推理過程，忽略語言]

[助理A的答案開始]
{answer_a}
[助理A的答案結束]

[助理B的答案開始]
{answer_b}
[助理B的答案結束]

請作為一個公正的裁判，評估下面顯示的用戶問題的兩個AI助理提供的回答的質量。您的評估應考慮正確性和有用性。注意，簡體中文不能使用，如果助理使用簡體中文要給最低分；文字必須符合台灣繁體中文。英文僅可用於專有名詞。您將獲得參考答案，助理A的答案，和助理B的答案。您的工作是評估哪個助理的答案更好。通過將兩個助理的答案與參考答案進行比較，開始您的評估。識別並更正任何錯誤。避免任何立場偏見，並確保回答的呈現順序不會影響您的決定。不要讓回答的長度影響您的評估。不要偏愛某個助理的名稱。請盡可能客觀。在提供解釋後，通過嚴格遵循以下格式輸出您的最終判決："[[A]]"如果助理A更好，"[[B]]"如果助理B更好，和"[[C]]"打成平手。"""


pairwise_score_eval_template = """[用戶問題]
{question}

[助理A的答案開始]
{answer_a}
[助理A的答案結束]

[助理B的答案開始]
{answer_b}
[助理B的答案結束]

請充分比較兩個AI助理對用戶問題的回答，並做出公正的判斷，以評估回答的質量。你的評估應該考慮因素，例如回答的有用性、相關性、準確性、深度、創造力和細節水平。注意，簡體中文不能使用，如果助理使用簡體中文要給最低分；文字必須符合台灣繁體中文。英文只接受專有名詞。確保回答的呈現順序不會影響你的決定。不要因回答的長度影響你的評估。不要偏愛某個助理的名稱。請盡量客觀。在提供解釋後，通過嚴格遵循以下格式輸出你的最終判決："A: 1~10分, B: 1~10分"。"""

pairwise_score_reference_eval_template = """[用戶問題]
{question}

[英文參考答案開始，請只考慮答案和推理過程，忽略語言]
{ref_answer_1}
[英文參考答案結束，請只考慮答案和推理過程，忽略語言]

[助理A的答案開始]
{answer_a}
[助理A的答案結束]

[助理B的答案開始]
{answer_b}
[助理B的答案結束]

請作為一個公正的裁判，評估下面顯示的用戶問題的兩個AI助理提供的回答的質量。您的評估應考慮正確性和有用性。注意，簡體中文不能使用，如果助理使用簡體中文要給最低分；文字必須符合台灣繁體中文。英文僅可用於專有名詞。您將獲得參考答案，助理A的答案，和助理B的答案。您的工作是評估哪個助理的答案更好。識別並更正任何錯誤。避免任何立場偏見，並確保回答的呈現順序不會影響您的決定。不要讓回答的長度影響您的評估。不要偏愛某個助理的名稱。請盡可能客觀。在提供解釋後，通過嚴格遵循以下格式輸出您的最終判決："A: 1~10分, B: 1~10分"。"""


def load_model_answer(path: str) -> dict[str, str]:
    df = pd.read_json(path, lines=True)
    if "question_id" in df.columns and "choices" in df.columns:
        return {
            row["question_id"]: row["choices"][0]["turns"][0]
            for _, row in df.iterrows()
        }
    elif "id" in df.columns and "generated_text" in df.columns:
        return {int(row["id"])+1: row["generated_text"] for _, row in df.iterrows()}
    else:
        raise ValueError("Unknown format")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("assistant_a_file", type=str)
    parser.add_argument("assistant_b_file", type=str)
    parser.add_argument("--judge_name", type=str, default="gpt-4")
    parser.add_argument("--question_path", type=str, default="zh_tw_bench/question.jsonl")
    parser.add_argument("--reference_path", type=str, default="zh_tw_bench/reference_answer/gpt-4.jsonl")
    parser.add_argument("--pk", action="store_true")
    args = parser.parse_args()

    questions = pd.read_json(args.question_path, lines=True)

    ref_answer = pd.read_json(args.reference_path, lines=True)

    a_name = Path(args.assistant_a_file).stem
    b_name = Path(args.assistant_b_file).stem
    assistant_a_answers = load_model_answer(args.assistant_a_file)
    assistant_b_answers = load_model_answer(args.assistant_b_file)

    for id in assistant_a_answers:
        assert id in questions["question_id"].values, f"{id} not in questions"
    for id in assistant_b_answers:
        assert id in questions["question_id"].values, f"{id} not in questions"

    is_pk = args.pk
    prompts = []
    for _, row in questions.iterrows():
        question_id = row["question_id"]
        question = row["turns"][0]
        answer_a = assistant_a_answers[question_id]
        answer_b = assistant_b_answers[question_id]
        category = row["category"]
        is_reference_based = category in NEED_REF_CATS
        if is_reference_based:
            ref_answer_1 = ref_answer[ref_answer["question_id"] == question_id]["choices"].values[0][0]["turns"][0]
            template = pairwise_pk_eval_template if is_pk else pairwise_score_reference_eval_template
            prompt = template.format(
                question=question, answer_a=answer_a, answer_b=answer_b, ref_answer_1=ref_answer_1
            )
            prompts.append({
                "A": a_name,
                "B": b_name,
                "category": category,
                "question_id": question_id,
                "prompt": prompt,
            })
            prompt = template.format(
                question=question, answer_a=answer_b, answer_b=answer_a, ref_answer_1=ref_answer_1
            )
            prompts.append({
                "A": b_name,
                "B": a_name,
                "category": category,
                "question_id": question_id,
                "prompt": prompt,
            })
        else:
            template = pairwise_pk_eval_template if is_pk else pairwise_score_eval_template
            prompt = template.format(
                question=question, answer_a=answer_a, answer_b=answer_b,
            )
            prompts.append({
                "A": a_name,
                "B": b_name,
                "category": category,
                "question_id": question_id,
                "prompt": prompt,
            })
            prompt = template.format(
                question=question, answer_a=answer_b, answer_b=answer_a,
            )
            prompts.append({
                "A": b_name,
                "B": a_name,
                "category": category,
                "question_id": question_id,
                "prompt": prompt,
            })
        print(prompt)
        print("----"*20)

    model_name = args.judge_name
    temperature = 0
    chat = ChatLiteLLM(temperature=temperature, model_name=model_name, max_retries=5)
    results = []
    for prompt in tqdm(prompts):
        print(f"A is {prompt['A']}, B is {prompt['B']}")
        result = chat.predict(prompt["prompt"])
        print(result)
        print("----"*20)
        prompt["result"] = result
        results.append(prompt)
        if args.pk:
            pd.DataFrame(results).to_excel(f"{a_name}_vs_{b_name}_pk.xlsx")
        else:
            pd.DataFrame(results).to_excel(f"{a_name}_vs_{b_name}_score.xlsx")
