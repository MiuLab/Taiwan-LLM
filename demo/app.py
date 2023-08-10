import time
import os
import gradio as gr
from text_generation import Client
from conversation import get_default_conv_template
from transformers import AutoTokenizer


endpoint_url = os.environ.get("ENDPOINT_URL", "http://127.0.0.1:8080")
client = Client(endpoint_url, timeout=120)
eos_token = "</s>"
max_new_tokens = 512
max_prompt_length = 4096 - max_new_tokens - 10

tokenizer = AutoTokenizer.from_pretrained("yentinglin/Taiwan-LLaMa-v1.0")

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        conv = get_default_conv_template("vicuna").copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # map human to USER and gpt to ASSISTANT
        for user, bot in history:
            conv.append_message(roles['human'], user)
            conv.append_message(roles["gpt"], bot)
        msg = conv.get_prompt()
        prompt_tokens = tokenizer.encode(msg)
        length_of_prompt = len(prompt_tokens)
        if length_of_prompt > max_prompt_length:
            msg = tokenizer.decode(prompt_tokens[-max_prompt_length+1:])

        history[-1][1] = ""
        for response in client.generate_stream(
                msg,
                max_new_tokens=max_new_tokens,
        ):
            if not response.token.special:
                character = response.token.text
                history[-1][1] += character
                yield history


    def generate_response(history, max_new_token=512, top_p=0.9, temperature=0.8, do_sample=True):
        conv = get_default_conv_template("vicuna").copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # map human to USER and gpt to ASSISTANT
        for user, bot in history:
            conv.append_message(roles['human'], user)
            conv.append_message(roles["gpt"], bot)
        msg = conv.get_prompt()

        for response in client.generate_stream(
                msg,
                max_new_tokens=max_new_token,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
        ):
            history[-1][1] = ""
            # if not response.token.special:
            character = response.token.text
            history[-1][1] += character
            print(history[-1][1])
            time.sleep(0.05)
            yield history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch()

#
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     with gr.Row():
#         with gr.Column(scale=4):
#             with gr.Column(scale=12):
#                 user_input = gr.Textbox(
#                     show_label=False,
#                     placeholder="Shift + Enter傳送...",
#                     lines=10).style(
#                     container=False)
#             with gr.Column(min_width=32, scale=1):
#                 submitBtn = gr.Button("Submit", variant="primary")
#         with gr.Column(scale=1):
#             emptyBtn = gr.Button("Clear History")
#             max_new_token = gr.Slider(
#                 1,
#                 1024,
#                 value=128,
#                 step=1.0,
#                 label="Maximum New Token Length",
#                 interactive=True)
#             top_p = gr.Slider(0, 1, value=0.9, step=0.01,
#                               label="Top P", interactive=True)
#             temperature = gr.Slider(
#                 0,
#                 1,
#                 value=0.5,
#                 step=0.01,
#                 label="Temperature",
#                 interactive=True)
#             top_k = gr.Slider(1, 40, value=40, step=1,
#                               label="Top K", interactive=True)
#             do_sample = gr.Checkbox(
#                 value=True,
#                 label="Do Sample",
#                 info="use random sample strategy",
#                 interactive=True)
#             repetition_penalty = gr.Slider(
#                 1.0,
#                 3.0,
#                 value=1.1,
#                 step=0.1,
#                 label="Repetition Penalty",
#                 interactive=True)
#
#     params = [user_input, chatbot]
#     predict_params = [
#         chatbot,
#         max_new_token,
#         top_p,
#         temperature,
#         top_k,
#         do_sample,
#         repetition_penalty]
#
#     submitBtn.click(
#         generate_response,
#         [user_input, max_new_token, top_p, top_k, temperature, do_sample, repetition_penalty],
#         [chatbot],
#         queue=False
#     )
#
#     user_input.submit(
#         generate_response,
#         [user_input, max_new_token, top_p, top_k, temperature, do_sample, repetition_penalty],
#         [chatbot],
#         queue=False
#     )
#
#     submitBtn.click(lambda: None, [], [user_input])
#
#     emptyBtn.click(lambda: chatbot.reset(), outputs=[chatbot], show_progress=True)
#
# demo.launch()