# TAME (TAiwan Mixture of Experts) <br/>LLM for Taiwanese Culture across Diverse Domains

<p align="center">
✍️ <a href="https://chat.twllm.com/" target="_blank">Online Demo</a>  
•
🤗 <a href="https://huggingface.co/collections/yentinglin/taiwan-llm-6523f5a2d6ca498dc3810f07" target="_blank">Model Collection</a> • 🐦 <a href="https://twitter.com/yentinglin56" target="_blank">Twitter/X</a> • 📃 <a href="https://arxiv.org/pdf/2311.17487.pdf" target="_blank">Model Paper</a> • 📃 <a href="https://arxiv.org/pdf/2403.20180" target="_blank">Eval Paper</a>  
• 👨️ <a href="https://yentingl.com/" target="_blank">Yen-Ting Lin</a> 
    <br/><br/>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/5df9c78eda6d0311fd3d541f/vlfv5sHbt4hBxb3YwULlU.png" width="500"> <br/>
    <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE">
<img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg"></a>
    <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE">
        <img src="https://img.shields.io/badge/Data_License-CC%20By%20NC%204.0-red.svg"></a>
    <br/>
Partnership with 和碩聯合科技, 長庚紀念醫院, 長春集團, 欣興電子, 律果, NVIDIA, 科技報橘   
</p>

# 🌟 [Demo Site](https://twllm.com/)

Try out Llama-3-Taiwan interactively at [twllm.com](https://twllm.com/)

# ⚔️ [Chatbot Arena](https://arena.twllm.com/)

Participate in the exciting [Chatbot Arena](https://arena.twllm.com/) and compete against other chatbots!

# 🚀 Quick Start for Fine-tuning

Using [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for fine-tuning:

```bash
# Run the axolotl docker image
docker run --gpus '"all"' --rm -it winglian/axolotl:main-latest

# Preprocess datasets (optional but recommended)
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess example_training_config_for_finetuning_twllm.yaml

# Fine-tune
accelerate launch -m axolotl.cli.train example_training_config_for_finetuning_twllm.yaml

```
Check out the example_training_config_for_finetuning_twllm.yaml file for detailed training configuration and parameters.
For more training framework information, visit [Axolotl's GitHub repository](https://github.com/OpenAccess-AI-Collective/axolotl).

--------


🚀 We're excited to introduce Llama-3-Taiwan-70B! Llama-3-Taiwan-70B is a 70B parameter model finetuned on a large corpus of Traditional Mandarin and English data using the Llama-3 architecture. It demonstrates state-of-the-art performance on various Traditional Mandarin NLP benchmarks.

The model was trained with [NVIDIA NeMo™ Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/) using the NVIDIA Taipei-1 built with [NVIDIA DGX H100](https://www.nvidia.com/en-us/data-center/dgx-h100/) systems.

The compute and data for training Llama-3-Taiwan-70B was generously sponsored by [Chang Gung Memorial Hospital](https://www.cgmh.org.tw/eng), [Chang Chun Group](https://www.ccp.com.tw/ccpweb.nsf/homepage?openagent), [Legalsign.ai](https://legalsign.ai/), [NVIDIA](https://www.nvidia.com/zh-tw/), [Pegatron](https://www.pegatroncorp.com/), [TechOrange](https://buzzorange.com/techorange/), and [Unimicron](https://www.unimicron.com/) (in alphabetical order).

We would like to acknowledge the [contributions](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct#contributions) of our data provider, team members and advisors in the development of this model, including [shasha77](https://www.youtube.com/@shasha77) for high-quality YouTube scripts and study materials, [Taiwan AI Labs](https://ailabs.tw/) for providing local media content, [Ubitus K.K.](https://ubitus.net/zh/) for offering gaming content, Professor Yun-Nung (Vivian) Chen for her guidance and advisement, Wei-Lin Chen for leading our pretraining data pipeline, Tzu-Han Lin for synthetic data generation, Chang-Sheng Kao for enhancing our synthetic data quality, and Kang-Chieh Chen for cleaning instruction-following data.


# Model Summary

Llama-3-Taiwan-70B is a large language model finetuned for Traditional Mandarin and English users. It has strong capabilities in language understanding, generation, reasoning, and multi-turn dialogue. Key features include:

- 70B parameters
- Languages: Traditional Mandarin (zh-tw), English (en)
- Finetuned on High-quality Traditional Mandarin and English corpus covering general knowledge as well as industrial knowledge in legal, manufacturing, medical, and electronics domains
- 8K context length
- Open model released under the Llama-3 license

# Training Details

- Training Framework: [NVIDIA NeMo](https://www.nvidia.com/zh-tw/ai-data-science/products/nemo/), [NVIDIA NeMo Megatron](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/megatron.html)
- Inference Framework: [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- Base model: [Llama-3 70B](https://llama.meta.com/llama3/)
- Hardware: [NVIDIA DGX H100](https://www.nvidia.com/zh-tw/data-center/dgx-h100/) on Taipei-1
- Context length: 8K tokens ([128k version](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-128k))
- Batch size: 2M tokens per step

# Evaluation

Checkout [Open TW LLM Leaderboard](https://huggingface.co/spaces/yentinglin/open-tw-llm-leaderboard) for full and updated list.

| Model                                                                            | [TMLU](https://arxiv.org/pdf/2403.20180) | Taiwan Truthful QA | [Legal Eval](https://huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1) |  [TW MT-Bench](https://huggingface.co/datasets/MediaTek-Research/TCEval-v2) | Long context | Function Calling | [TMMLU+](https://github.com/iKala/ievals) | 
|---------------------------------------------------------------------------------|--------------|---------------|--------------------|--------------|--------------|-----------------|-----------| 
|      | 學科知識 | 台灣在地化測試 | 台灣法律考題 |  中文多輪對答 | 長文本支援 | 函數呼叫 |  | 
| [**yentinglin/Llama-3-Taiwan-70B-Instruct**](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct)     | **74.76%**       |     80.95%          |      68.42%              |      7.54        |    [128k version](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-128k)          |        ✅         |     67.53%      |
| [**yentinglin/Llama-3-Taiwan-70B-Instruct-DPO**](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-DPO)     | 74.60%       |     **81.75%**          |      **70.33%**              |      -       |    -   |        ✅         | - |
| [**yentinglin/Llama-3-Taiwan-70B-Instruct-128k**](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-128k)     | 73.01%       |     80.16%          |      63.64%              |   -    |   -  |        ✅         |  -  |
| [**yentinglin/Llama-3-Taiwan-8B-Instruct**](https://huggingface.co/yentinglin/Llama-3-Taiwan-8B-Instruct) | 59.50%       |    61.11%           |         53.11%           |     7.21         |     [128k version](https://huggingface.co/yentinglin/Llama-3-Taiwan-8B-Instruct-128k)         |        ✅         |    52.28%       |
| [**yentinglin/Llama-3-Taiwan-8B-Instruct-DPO**](https://huggingface.co/yentinglin/Llama-3-Taiwan-8B-Instruct-DPO) | 59.88%       |    59.52%           |         52.63%           |  -  |  -  |        ✅         |  -   |
| [**yentinglin/Llama-3-Taiwan-8B-Instruct-128k**](https://huggingface.co/yentinglin/Llama-3-Taiwan-8B-Instruct-128k) | -  |  -   |    -     | - | -  |        ✅         |  -  |
| [Claude-3-Opus](https://www.anthropic.com/api) | [73.59% (5-shot)](https://arxiv.org/pdf/2403.20180)       |  [69.84%](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-rc3/tree/main/opus-Taiwan-Truthful-QA)    |     [60.29%](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-rc3/tree/main/opus)      |       -       |      200k       |        ✅         |     -      |
| [GPT4-o](https://platform.openai.com/docs/api-reference/chat/create) | [65.56% (0-shot), 69.88% (5-shot)](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-rc3/tree/main/4o-tmlu) | [76.98%](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-rc3/tree/main/4o-Taiwan-Truthful-QA)  |    [53.59%](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-rc3/tree/main/4o)   | -  |      128k        |        ✅         |    -  |
| [GPT4-turbo](https://platform.openai.com/docs/api-reference/chat/create) | [70.42% (5-shot)](https://arxiv.org/pdf/2403.20180)       |        -       |              -      |          -   |      128k        |        ✅         |     60.34%^      |
| [Gemini-Pro](https://ai.google.dev/gemini-api/docs) | [61.40% (5-shot)](https://arxiv.org/pdf/2403.20180)       |          -     |            -        |     -         |       1000k       |        ✅         |    49.92%^     |
| [GPT-3.5-turbo-1106](https://platform.openai.com/docs/api-reference/chat/create) | [49.37% (5-shot)](https://arxiv.org/pdf/2403.20180)       |        -       |         -           |    7.1         |      128k        |        ✅         |     	41.76%^      |
| [Qwen1.5-110B-Chat](https://huggingface.co/Qwen/Qwen1.5-110B-Chat)                                                         | **75.69%**       |   66.67%    |    49.28%                |     -         |       32k       |        ✅         |    65.81%      |
| [Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat)                                                              | 73.59%       |    71.43%           |          55.02%          |      6.9        |      200k        |        ✅         |      64.10%     |
| [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)                                           | 70.95%       |       65.08%        |       52.63%             |      -        |       8k       |        ✅         |   62.75%        |
| [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)                                     | 55.57%       |     52.38%          |      44.98%              |      -        |       64k       |        ✅         |         52.16%  |
| [Breexe-8x7B-Instruct-v0_1](https://huggingface.co/MediaTek-Research/Breexe-8x7B-Instruct-v0_1)     | -       |      -         |           -         |      7.2        |      8k        |        ❓         |     48.92%      |
| [c4ai-command-r-plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus)                                                | 62.87%       |      64.29%         |         34.45%           |         -     |         128k     |        ✅         |      49.75%     |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)                                            | 55.81%       |     46.83%          |     35.89%               |       -       |        8k      |        ✅         |       43.38%    |
| [Breeze-7B-Instruct-v1_0](https://huggingface.co/MediaTek-Research/Breeze-7B-Instruct-v1_0)                                      | 55.57%       |     52.38%          |      39.23%              |     6.0         |      32k        |        ❓         |       41.77%    |
| [Llama3-TAIDE-LX-8B-Chat-Alpha1](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)                                           | 47.30%       |  50.79%             |   37.80%                 |       -       |     8k         |        ❓         |      39.03%     |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)                                               | 40.97%       |     37.30%          |      27.27%              |      -        |   4k           |        ❓         |        33.02%   |

Numbers are 0-shot by default.

[Eval implementation](https://github.com/adamlin120/lm-evaluation-harness)

^ taken the closet matching numbers from original dataset.

## Needle in a Haystack Evaluation

The "Needle in a 出師表" evaluation tests the model's ability to locate and recall important information embedded within a large body of text, using the classic Chinese text 《出師表》 by 諸葛亮.

To run the evaluation, use the [script](https://github.com/adamlin120/needle-haystack/tree/main).


# TW MT-Bench Score

- Average Score: 7.5375
- Maximum Score: 10
- Minimum Score: 1
- Median Score: 9.0
- Standard Deviation: 3.0349783771882133
- Total Number of Scores: 160
- [Model response](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-rc1/blob/main/Llama-3-Taiwan.jsonl)
- [GPT-4 Eval](https://huggingface.co/yentinglin/Llama-3-Taiwan-70B-Instruct-rc1/blob/main/gpt-4_single.jsonl)
- [Code fork from `mtkresearch/TCEval` with bug fixing](https://github.com/adamlin120/TCEval)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5df9c78eda6d0311fd3d541f/hfstnJElgB39Um4Bco11D.png)


# Use Cases

Llama-3-Taiwan-70B can be applied to a wide variety of NLP tasks in Traditional Mandarin and English, including:

<details>
<summary>1. 多輪對話</summary>

<pre>
System: You are an AI assistant called Twllm, created by TAME (TAiwan Mixture of Expert) project.

User: 嗨，你好！

Assistant: 你好！我今天可以如何協助你？

User: 就想聊天而已....

Assistant: 當然，我很樂意和你聊天！有什麼特別想談論的嗎？
</pre>

</details>

<details>
<summary>2. RAG（檢索增強生成）</summary>

Demo: 可以打開 `Search Web` on [twllm.com](https://twllm.com/)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5df9c78eda6d0311fd3d541f/Zcl20NsPmY2vObIaMtT8G.png)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5df9c78eda6d0311fd3d541f/xuXEHIkMgDEB7xxDd3JoL.png)

<pre>
System: You are an AI assistant called Twllm, created by TAME (TAiwan Mixture of Expert) project.

User: 明天台大天氣如何？

Assistant: 根據天氣預報，明天台大天氣為陣雨。低溫 24°C。10 到 15 公里/小時 的 東 風 。降雨機率為 40%。
</pre>

</details>

<details>
<summary>3. 格式化輸出, 語言理解, 實體識別, 函數調用</summary>

If you are interested in function-calling, I strongly recommend [using constrained decoding](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance) to turn on `json mode`.

Example from HW7 in `INTRODUCTION TO GENERATIVE AI 2024 SPRING` from `HUNG-YI LEE (李宏毅)`


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5df9c78eda6d0311fd3d541f/dCGo92llL3_ammH_Nu2-c.png)

<pre>
System: You are an AI assistant called Twllm, created by TAME (TAiwan Mixture of Expert) project.

User: 您是電影評論分析師，負責情緒分析。
分析每個被評論的句子，評估句子中每個字和標點符號的重要性，
並且將結果使用 tuple 的結構列出。
tuple 值的範圍介於 -1 (負向) 到 1 (正向) 之間，
然後，將評論分類為正向 (1) 或負向 (-1)。
這些需要被評論的句子，包含在 <review> 標籤中間。

輸出範例:
[(<單字或標點符號>, <float importance>), ...]
<int classification>
<review> 根據外媒報導，宏都拉斯在去年3月與台灣斷交，並轉往與中國建交之後，卻遲遲等不到中國的金援，談好要進口宏都拉斯的白蝦也拖延許久，甚至還將白蝦價格殺價至過往與台灣交易的半價，再加上墨西哥同時也中斷了與宏國的白蝦貿易，對當地蝦農可說是雪上加霜。 <review>
  

Assistant: [('根據', 0.2), ('外媒', 0.1), ('報導', -0.1), (',', 0.05), ('宏都拉斯', -0.3), ('在去年3月與台灣斷交， ', -0.5), ('並轉往與中國建交之後， ', -0.4), ('卻遲遲等不到中國的金援， ', -1),
('談好要進口宏都拉斯的白蝦也拖延許久， ', -1),
(',甚至還將白蝦價格殺價至過往與台灣交易的半價，再加上墨西哥同時也中斷了與宏國的白蝦貿易，對當地蝦農可說是雪上加霜。 ',-1)]
-1

</pre>
</details>





# Get Started

*Caveat: Set these as stop tokens: ["USER:", "ASSISTANT:", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]*

## Hugging Face Transformers library
You can use Llama-3-Taiwan-70B with the Hugging Face Transformers library:


```python
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "system", "content": "You are an AI assistant called Twllm, created by TAME (TAiwan Mixture of Expert) project."},
    {"role": "user", "content": "你好，請問你可以完成什麼任務？"},
    {"role": "assistant", "content": "你好，我可以幫助您解決各種問題、提供資訊並協助完成多種任務。例如：回答技術問題、提供建議、翻譯文字、尋找資料或協助您安排行程等。請告訴我如何能幫助您。"},
    {"role": "user", "content": "太棒了！"},
]
pipe = pipeline("text-generation", model="yentinglin/Llama-3-Taiwan-70B-Instruct")
pipe(messages)
```

## vLLM

Start the server
```bash
export NUM_GPUS=4
export PORT=8000

docker run \
  -e HF_TOKEN=$HF_TOKEN \
  --gpus '"device=0,1,2,3"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p "${PORT}:8000" \
  --ipc=host \
  vllm/vllm-openai:v0.4.0.post1 \
  --model "yentinglin/Llama-3-Taiwan-70B-Instruct" \
  -tp "${NUM_GPUS}"
```

Sample client code, or you can use anything OpenAI-API compatible clients

```python
# pip install "openai>=1.0.0"
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="yentinglin/Llama-3-Taiwan-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)
```


Enjoy exploring the capabilities of Llama-3-Taiwan-70B! We look forward to seeing what you create with this powerful open-source model. If you have any questions or feedback, please let us know.

# Citation
```
@article{DBLP:journals/corr/abs-2311-17487,
  author       = {Yen{-}Ting Lin and
                  Yun{-}Nung Chen},
  title        = {Taiwan {LLM:} Bridging the Linguistic Divide with a Culturally Aligned
                  Language Model},
  journal      = {CoRR},
  volume       = {abs/2311.17487},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2311.17487},
  doi          = {10.48550/ARXIV.2311.17487},
  eprinttype    = {arXiv},
  eprint       = {2311.17487},
  timestamp    = {Tue, 05 Dec 2023 14:40:42 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2311-17487.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@article{DBLP:journals/corr/abs-2403-20180,
  author       = {Po{-}Heng Chen and
                  Sijia Cheng and
                  Wei{-}Lin Chen and
                  Yen{-}Ting Lin and
                  Yun{-}Nung Chen},
  title        = {Measuring Taiwanese Mandarin Language Understanding},
  journal      = {CoRR},
  volume       = {abs/2403.20180},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2403.20180},
  doi          = {10.48550/ARXIV.2403.20180},
  eprinttype    = {arXiv},
  eprint       = {2403.20180},
  timestamp    = {Wed, 10 Apr 2024 17:37:45 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2403-20180.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Previous Taiwan-LLM Releases

The Taiwan LLM Initiative was started by Yenting Lin (林彥廷) in July 2023.

- Version 1.0 was released in August 2023.
- Version 2.0 was released in October 2023, sponsored by Ubitus K.K.

These models are designed to support Traditional Mandarin and are optimized for Taiwanese culture and related applications. For more detailed information about our models, including demos, features, and examples, please visit our [Hugging Face collection](https://huggingface.co/collections/yentinglin/taiwan-llm-6523f5a2d6ca498dc3810f07).


# Disclaimer

This model is provided “as‑is” and without warranties of any kind. Users are solely responsible for evaluating the accuracy and suitability of the outputs. The developers assume no liability for any direct or indirect damages arising from its use.  
The model is strictly not intended for high‑risk applications such as medical diagnosis, legal advice, or financial investment. For such use cases, please consult qualified professionals.

本模型「如是」（as‑is）提供，使用者須自行評估結果之正確性與適用性。開發者對於使用本模型所引發之任何直接或間接損失，不承擔任何法律責任。  
嚴禁用於醫療診斷、法律諮詢、金融投資等高風險場景；若有相關需求，請尋求專業人員協助。
