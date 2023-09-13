# Language Models for Taiwanese Culture 


<p align="center">
✍️ <a href="https://huggingface.co/spaces/yentinglin/Taiwan-LLaMa2" target="_blank">Online Demo</a>  
•
🤗 <a href="https://huggingface.co/yentinglin" target="_blank">HF Repo</a> • 🐦 <a href="https://twitter.com/yentinglin56" target="_blank">Twitter</a> • 📃 <a href="https://arxiv.org/pdf/2305.13711.pdf" target="_blank">[Paper Coming Soon]</a>  
• 👨️ <a href="https://yentingl.com/" target="_blank">Yen-Ting Lin</a> 
    <br/><br/>
    <img src="https://www.csie.ntu.edu.tw/~miulab/taiwan-llama/logo-v2.png" width="100"> <br/>
    <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE">
<img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg"></a>
    <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE">
        <img src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg"></a>
    <br/>
   
</p>




## Overview
Taiwan-LLaMa is a full parameter fine-tuned model based on LLaMa 2 for Traditional Mandarin applications.

**Taiwan-LLaMa v1.0** pretrained on over 5 billion tokens and instruction-tuned on over 490k conversations both in traditional mandarin.

## Demo
A live demonstration of the model can be accessed at [Hugging Face Spaces](https://huggingface.co/spaces/yentinglin/Taiwan-LLaMa2).

## Key Features

1. **Traditional Mandarin Support**: The model is fine-tuned to understand and generate text in Traditional Mandarin, making it suitable for Taiwanese culture and related applications.

2. **Instruction-Tuned**: Further fine-tuned on conversational data to offer context-aware and instruction-following responses.

3. **Performance on Vicuna Benchmark**: Taiwan-LLaMa's relative performance on Vicuna Benchmark is measured against models like GPT-4 and ChatGPT. It's particularly optimized for Taiwanese culture.

4. **Flexible Customization**: Advanced options for controlling the model's behavior like system prompt, temperature, top-p, and top-k are available in the demo.


## Work in progress
- [ ] **Improved pretraining**: A refined pretraining process (e.g. more data from Taiwan, training strategies) is under development, aiming to enhance model performance for better Taiwanese culture.
- [ ] **Extend max length**: Utilizing the Rope mechanism as described in [the paper](https://arxiv.org/abs/2104.09864), the model's length will be extended from 4k to 8k.


## Taiwanese Culture Examples

| Question  | ChatGPT  | **Taiwan LLaMa v1.0**  | **Taiwan LLaMa v0.0 (w/o pretraining)**  | Llama2_Traditional_Chinese_13b_Chat  | 
|:-|:-|:-|:-|:-|
| 以惡地景觀著名的是高雄哪一個景點?   | 以惡地地景著名的是高雄市的旗津。    | 高雄以擁有惡地景觀著名的景點是田寮月世界。✅ | 以惡地景觀著名的是高雄市的旗津島。旗津島是一個小島，... | 高雄市內有多個以惡地景觀著名的景點，其中以下面幾個最為著名： 1. 高雄捷運紅線沿途景點：高雄市政府為了推廣高雄  |
| 唐鳳是誰？  | 唐鳳（Audrey Tang）是台灣的政治家、社會活動家和軟體工程師。她於2016年擔任台灣行政院數位政務委員，是台灣政府歷史上首位沒有政黨背景的政務官。唐鳳在軟體工程領域有豐富的經驗。 ✅ | 唐鳳是政治評論家、資訊社會學研究者，也是台灣第一位擔任政府數位政委的人。 ✅ | 唐鳳（Tang Feng）是一位著名的中國演員和歌手。他出生於1974年12月17日，出生地點在中國北京。唐鳳在1995年開始他的演藝生涯，並在同年發行了他的首張專輯《我不孤單》。此後，他發行了許多受歡迎的歌曲，並在多部電視劇和電影中擔任主角。唐鳳因其精湛的演技和迷人的歌聲而被譽為“亞洲魅力男王”。他的音樂獲得了多個獎項，包括“華表最佳新男演唱獎”、“華表最佳歌曲獎”和“華表最佳舞蹈獎”等。  | 唐鳳是唐朝時期的一位著名文學家，他的代表作品包括《陰騭》、《狂凡子》等。他的文學風格具有強烈的個性和創意，被譽為「唐代文學之父」。  |
| NTU 在哪?                      | NTU位於新加坡，具體地址是50 Nanyang Avenue, Singapore 639798。   | NTU 是國立臺灣大學的縮寫。它是臺灣的一所頂尖研究型大學，位於臺北市。 ✅| NTU 是台灣的一所大學，位於台北市。✅ | NTU 的總部位於台灣新北市板橋區，佔地面積約為1,000公頃。|
| 甚麼是22K? | "22K" 可能指的是金的純度，尤其是在珠寶和金飾方面。 | "22K"是指台灣的大學畢業生的起薪，大約是新台幣22,000元。這個詞被用來形容大學畢業生的起薪偏低，並且在就業市場上對於大學畢業生的需求不足。✅| 22K是指溫度尺度上的22,000K（千度）。1K代表一度，即溫度差距，是攝氏溫標（攝氏度）的一百萬分之一。因此，22K等於-253.15°C（攝氏溫標）或-423.67°F（華氏溫標）。 |  22K是一種簡寫，即22,000。它是一個數字，表示22,000個單位。|
| 肉圓要不要加香菜？                  | 這是個個人口味的問題，有些人喜歡在肉圓中加入香菜，因為香菜的清香可以增添風味；而有些人則不喜歡香菜的味道，覺得它會蓋過其他食材的味道。所以要不要加香菜，可以根據個人喜好來決定。 | 這是一個個人喜好的問題，有些人喜歡在肉圓裡加香菜，有些人不喜歡。**在台灣**，肉圓加香菜是一種常見的做法，這是一種在肉圓上加香菜的做法。然而，在其他國這是一個文化差異的問題。😊 | 肉圓加香菜可以增添一些芳香和美味的口感，如果你喜歡香菜的味道，可以試試加入肉圓中。香菜通常在製作過程中炒香，可以增添一些芳香和美味的味道。如果你想要加入香菜，可以在製作過程中炒香一些香菜，然後將其混入肉圓中。肉圓加香菜可以增添一些芳香和美味的口感，如果你喜歡香菜的味道，可以試試加入肉圓中。| 加香菜的肉圓味道更好，但是不加香菜的肉圓也很好吃。如果你喜歡香菜的味道，那麼加上香菜就好了。如果你不喜歡香菜，那麼不加也不會影響肉 |


## Model

We provide a number of model checkpoints that we trained. Please find them on Hugging Face [here](https://huggingface.co/models?search=taiwan-llama). Here are some quick links to the checkpoints that are finetuned from LLaMa 2:

| **Model**                                              | **13B**                                                                                                                       | 
|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Taiwan-LLaMa v1.0** (_better for Taiwanese Culture_)   | 🤗 <a href="https://huggingface.co/yentinglin/Taiwan-LLaMa-v1.0" target="_blank">yentinglin/Taiwan-LLaMa-v1.0</a>  | 
| Taiwan-LLaMa v0.9 (partial instruction set)            | 🤗 <a href="https://huggingface.co/yentinglin/Taiwan-LLaMa-v0.9" target="_blank">yentinglin/Taiwan-LLaMa-v0.9</a>  | 
| Taiwan-LLaMa v0.0 (no Traditional Mandarin pretraining) | 🤗 <a href="https://huggingface.co/yentinglin/Taiwan-LLaMa-v0.0" target="_blank">yentinglin/Taiwan-LLaMa-v0.0</a>  | 

## Data

經過專業律師的法律建議，我們暫時將有版權疑慮的資料集下架。我們會積極尋求更多的法律意見，希望能儘快重新上架這些訓練資料集。

Due to legal concerns raised by our legal advisors regarding copyrighted material, we've temporarily removed the datasets. We're actively seeking further opinions and hope to make the training datasets available again soon.

| **Dataset**                     | **Link**                                                                                                                      | 
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| ~~**Instruction-tuning**~~      | 🤗 <a href="https://huggingface.co/datasets/yentinglin/traditional_mandarin_instructions" target="_blank">yentinglin/traditional_mandarin_instructions</a>                                           | 
| ~~Traditional Mandarin Pretraining~~ | 🤗 <a href="https://huggingface.co/datasets/yentinglin/zh_TW_c4" target="_blank">yentinglin/zh_TW_c4</a>                                   | 

## Architecture
Taiwan-LLaMa is based on LLaMa 2, leveraging transformer architecture, <a href="https://github.com/Dao-AILab/flash-attention" target="_blank">flash attention 2</a>, and bfloat16.

It includes:

* Pretraining Phase: Pretrained on a vast corpus of over 5 billion tokens, extracted from common crawl in Traditional Mandarin.
* Fine-tuning Phase: Further instruction-tuned on over 490k multi-turn conversational data to enable more instruction-following and context-aware responses.

## Evaluating "Taiwan LLaMa" on TTQA (Taiwan Trivia Question Answering)

[Link to the Origianl TTQA dataset from MediaTek Research](https://github.com/mtkresearch/MR-Models/tree/main/datasets/TTQA)

[Link to the Taiwan LLaMa and ChatGPT Prediction on  TTQA](https://github.com/MiuLab/Taiwan-LLaMa/blob/main/evaluation/TTQA_1.0.0_tw_llama_v1.0.json)

Taiwan LLaMa is being benchmarked on the TTQA dataset, a evaluation set for question answering models specific to Taiwan-related trivia.

As of the current stage, Taiwan LLaMa has achieved a score identical to 75% (48 out of 64) of the performance observed with ChatGPT. (Both in 0-shot)

Further comparisons and in-depth analysis are in progress.


## Generic Capabilities on Vicuna Benchmark 

The data is translated into traditional mandarin for evaluating the general capability.


<img src="./images/zhtw_vicuna_bench_chatgptbaseline.png" width="700">

The scores are calculated with ChatGPT as the baseline, represented as 100%. The other values show the relative performance of different models compared to ChatGPT.

| Language Model                      | Relative Score (%) |
|-------------------------------------|--------------------|
| GPT-4                               | 102.59%            |
| ChatGPT                             | 100.00%            |
| **Taiwan-LLaMa v1.0**               | 76.76%             |
| Claude-Instant-1.2                  | 74.04%             |
| Llama2_Traditional_Chinese_13b_Chat | 56.21%             |




## How to deploy the model on my own machine?
We recommend hosting models with [🤗 Text Generation Inference](https://github.com/huggingface/text-generation-inference). Please see their [license](https://github.com/huggingface/text-generation-inference/blob/main/LICENSE) for details on usage and limitations.
```bash
bash run_text_generation_inference.sh "yentinglin/Taiwan-LLaMa-v1.0" NUM_GPUS DIR_TO_SAVE_MODEL PORT MAX_INPUT_LEN MODEL_MAX_LEN
```

Prompt format follows vicuna-v1.1 template:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {user} ASSISTANT:
```

## Setup development environment
```bash
conda create -n taiwan-llama python=3.10 -y 
conda activate taiwan-llama
pip install -r requirements.txt
```

## 常見問顔

### 網頁Demo與程式碼執行結果有落差

可以參考這個 [#19 (comment)](https://github.com/your_repository/issues/19#issuecomment-YourCommentID)。

### 能否商用化？

關於模型能不能商用，我建議您自行尋求法律意見。

模型作者 (Meta 與我) 都願意開放商用，但是『可以商用的模型”訓練在“有著作權法保護的資料上”，是否可以商用』需要您的判斷。

台灣沒有相關法案保護模型訓練在有著作權的資料上，但就我的理解，我們模型雖訓練在著作權資料上，但並沒有抄襲著作權人的意思表示，所以模型是可以商用的。

以上是我諮詢律師的結論，為求謹慎還請您尋求更專業的法律意見。

### 請問訓練此模型時使用的機器規格

Pretraining: 8 x A100 80G for 2 weeks  
Instruction finetuning: 8 x H100 for 12 hrs

#### English Version

### Web Demo and Code Execution Results Differ

Refer to this [#19 (comment)](https://github.com/your_repository/issues/19#issuecomment-YourCommentID).

### Can it be Commercialized?

For questions on commercial use, consult legal advice.

Both the model authors (Meta and I) are open to commercial use. However, whether a "commercially usable model" trained on "copyrighted data" can be used commercially is for you to decide.

To my understanding, although the model is trained on copyrighted data, it does not plagiarize. Therefore, it can be commercialized.

This is based on legal advice; for caution, consult a legal expert.

### Machine Specifications for Training This Model

Pretraining: 8 x A100 80G for 2 weeks  
Instruction finetuning: 8 x H100 for 12 hrs

## Citations
If you use our code, data, or models in your research, please cite this repository. You can use the following BibTeX entry:

```bibtex
@inproceedings{lin-chen-2023-llm,
    title = "{LLM}-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models",
    author = "Lin, Yen-Ting  and Chen, Yun-Nung",
    booktitle = "Proceedings of the 5th Workshop on NLP for Conversational AI (NLP4ConvAI 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.nlp4convai-1.5",
    pages = "47--58"
}

@misc{taiwanllama,
    author={Lin, Yen-Ting and Chen, Yun-Nung},
    title={Language Models for Taiwanese Culture},
    year={2023},
    url={https://github.com/MiuLab/Taiwan-LLaMa},
    note={Code and models available at https://github.com/MiuLab/Taiwan-LLaMa},
}
```

## Collaborate With Us
If you are interested in contributing to the development of Traditional Mandarin language models, exploring new applications, or leveraging Taiwan-LLaMa for your specific needs, please don't hesitate to contact us. We welcome collaborations from academia, industry, and individual contributors.

## License
The code in this project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

The models included in this project are licensed under the LLAMA 2 Community License. See the [LLAMA2 License](https://github.com/facebookresearch/llama/blob/main/LICENSE) for full details.

## OpenAI Data Acknowledgment
The data included in this project were generated using OpenAI's models and are subject to OpenAI's Terms of Use. Please review [OpenAI's Terms of Use](https://openai.com/policies/terms-of-use) for details on usage and limitations.


## Acknowledgements

We thank [Meta LLaMA team](https://github.com/facebookresearch/llama) and [Vicuna team](https://github.com/lm-sys/FastChat) for their open-source efforts in democratizing large language models.
