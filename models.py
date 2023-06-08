import openai
import torch

def BERT(model, tokenizer, input):
    inputs = tokenizer(input, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    if predicted_class_id==1:
        return 'Positive Sentiment'
    else:
        return 'Negative Sentiment'

def GPT(input):
    messages = [
        {
            'role': 'user',
            'content': 'Now I will give you a sentence, you should tell me that it contains positive sentiments or negative sentiments in the first line and tell me the reason.\n Sentence: {}'.format(input)
        }
    ]

    generation_config = dict(
        temperature=1,
        top_p=1,
        n=1, # n>1 is not supported in my default OpenAITask class. Please write your own task class if you want it.
        max_tokens = 1024,
    )

    openai.api_key = 'your key'
    openai.api_base = 'your base'
    openai.api_type = 'your type'
    openai.api_version = 'your version'

    completion = openai.ChatCompletion.create(
        engine='your engine',
        messages=messages
    )

    return completion['choices'][0]['message']['content']