# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import pandas as pd

from langchain.chat_models import ChatAnthropic


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to generate the answer.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output file to store the generated answers.",
    )
    args = parser.parse_args()

    claude = ChatAnthropic(model_name=args.model_name)

    question_df = pd.read_json("zh_tw_bench/question.jsonl", lines=True)
    dump = []

    for i, row in question_df.iterrows():
        question = row['turns'][0]
        print(f"Question: {question}")
        generated_text = claude.predict(question, temperature=0.7, max_length=1024)
        print(f"Answer: {generated_text}")
        dump.append({
            "id": i,
            "generated_text": generated_text,
            "model_name": args.model_name,
        })
    pd.DataFrame(dump).to_json(args.output_file, orient="records", lines=True)
