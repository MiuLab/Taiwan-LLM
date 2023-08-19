import json
import re
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from langchain.chat_models import ChatLiteLLM


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
    )
    parser.add_argument(
        "output_file",
        type=str,
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input_file)

    scores_ab = defaultdict(list)

    chat = ChatLiteLLM(model_name='gpt-3.5-turbo')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text_score = row['result']

        scores = re.findall(r'(A|B): (\d+)分', text_score)
        result = {score[0]: int(score[1]) for score in scores}

        print(result)  # Output: {'A': 10, 'B': 7}
        if result:
            scores_ab[row['A']].append(result['A'])
            scores_ab[row['B']].append(result['B'])
            # print sum scores
            print(row['A'], sum(scores_ab[row['A']]))
            print(row['B'], sum(scores_ab[row['B']]))
        else:
            print(f"Cannot parse score")
            print("Using human input")
            print()
            print(text_score)


            a_score = int(input("What is the score for A?"))
            b_score = int(input("What is the score for B?"))
            scores_ab[row['A']].append(a_score)
            scores_ab[row['B']].append(b_score)
            # print sum scores
            print(row['A'], sum(scores_ab[row['A']]))
            print(row['B'], sum(scores_ab[row['B']]))
            continue
    with open(args.output_file, 'w') as f:
        json.dump(scores_ab, f, indent=4)


    #
    #         a_name = row['A']
    #         b_name = row['B']
    #         prompt = f"""text: {text_score}
    #
    # Given the text containing score for A and B.
    # Output the score for A and B in json format.
    #     "A": int,
    #     "B": int
    # """
    #         success = False
    #         temperature = 0.0
    #         while not success:
    #             try:
    #                 json_text = chat.predict(prompt, temperature=temperature)
    #                 # print(json_text)
    #                 parsed_score = json.loads(json_text)
    #                 # print(parsed_score)
    #                 scores_ab[a_name].append(parsed_score['A'])
    #                 scores_ab[b_name].append(parsed_score['B'])
    #                 # print sum scores
    #                 print(a_name, sum(scores_ab[a_name]))
    #                 print(b_name, sum(scores_ab[b_name]))
    #                 success = True
    #                 with open(args.output_file, 'w') as f:
    #                     json.dump(scores_ab, f, indent=4)
    #             except Exception as e:
    #                 print(e)
    #                 temperature += 0.1
    #                 continue
