# -*- coding: utf-8 -*-
import time
from argparse import ArgumentParser

import pandas as pd

from conversation import get_conv_template

from vllm import LLM, SamplingParams


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--prompt_template",
        type=str,
        required=True,
        help="The template to generate the answer.",
    )
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

    question_df = pd.read_json("zh_tw_bench/question.jsonl", lines=True)
    prompts = []

    for i, row in question_df.iterrows():
        question = row['turns'][0]
        print(f"Question: {question}")
        conv = get_conv_template(args.prompt_template)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f"Prompt: {prompt}")
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
    llm = LLM(model=args.model_name, tensor_parallel_size=2)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        id = output.request_id
        generated_text = output.outputs[0].text
    dump = [
        {
            "id": output.request_id,
            "generated_text": output.outputs[0].text,
            "model_name": args.model_name,
        }
        for output in outputs
    ]
    pd.DataFrame(dump).to_json(args.output_file, orient="records", lines=True)
