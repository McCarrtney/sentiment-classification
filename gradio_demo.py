import sys
import gradio as gr
import argparse
import os
import mdtex2html
from model import GPT, BERT, TFIDF_KNN, TFIDF_LR, BERT_KNN, BERT_LR

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], []

def transform(label):
    return "Positive Sentiment" if label==1 else "Negative Sentiment"

def predict(
    input,
    chatbot,
    history,
    model,
    **kwargs,
):
    # start = time.time()
    now_input = input
    chatbot.append((input, ""))
    history = history or []

    if model=='GPT-3.5-turbo':
        output = GPT(input)
    elif model=='BERT (for classification)':
        output = transform(BERT(input))
    elif model=="Logistic Regression (TFIDF)":
        output = transform(TFIDF_LR(input))
    elif model=="Logistic Regression (BERT)":
        output = transform(BERT_LR(input))
    elif model=="KNN (TFIDF)":
        output = transform(TFIDF_KNN(input))
    elif model=="KNN (BERT)":
        output = transform(BERT_KNN(input))

    history.append((now_input, output))
    chatbot[-1] = (now_input, output)
    yield chatbot, history
    

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Data Mining Project</h1>""")
    with gr.Column(elem_id="col_container"):
        model = gr.Radio(
            value="Logistic Regression (TFIDF)",
            choices=[
                "Logistic Regression (TFIDF)",
                "Logistic Regression (BERT)",
                "KNN (TFIDF)",
                "KNN (BERT)",
                "BERT (for classification)",
                "GPT-3.5-turbo",
            ],
            label="Model",
            interactive=True,
        )

        chatbot = gr.Chatbot(elem_id="chatbot")
        inputs = gr.Textbox(
            placeholder="Hi there!", label="Type an input and press Enter"
        )
        with gr.Row(elem_id="col_container"):
            b1 = gr.Button('Run')
            emptyBtn = gr.Button("Clear History")

    history = gr.State([])  # (message, bot_message)
    
    b1.click(
        predict,
        [inputs, chatbot, history, model],
        [chatbot, history],
        show_progress=True
    )
    b1.click(reset_user_input, [], [inputs])
    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
