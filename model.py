import openai
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from config import *
import joblib

def TFIDF_LR(input):
    vectorizer = joblib.load(TFIDF_VECTOR_PATH)
    a = vectorizer.transform([input])

    lr = joblib.load(TFIDF_LR_PATH)
    pred = lr.predict(a)
    return pred[0]

def TFIDF_KNN(input):
    vectorizer = joblib.load(TFIDF_VECTOR_PATH)
    a = vectorizer.transform([input])

    knn = joblib.load(TFIDF_KNN_PATH)
    pred = knn.predict(a)
    return pred[0]

def BERT_LR(input):
    tokenizer = BertTokenizer.from_pretrained(BERT_VECTOR_PATH)
    model = BertModel.from_pretrained(BERT_VECTOR_PATH)

    with torch.no_grad():
        encoded_input = tokenizer(input, return_tensors='pt')
        encoded_input = {k:v for k, v in encoded_input.items()}
        output = model(**encoded_input)
        x = output.last_hidden_state[:, 0, :].numpy()

    lr = joblib.load(BERT_LR_PATH)
    pred = lr.predict(x)
    return pred[0]

def BERT_KNN(input):
    tokenizer = BertTokenizer.from_pretrained(BERT_VECTOR_PATH)
    model = BertModel.from_pretrained(BERT_VECTOR_PATH)

    with torch.no_grad():
        encoded_input = tokenizer(input, return_tensors='pt')
        encoded_input = {k:v for k, v in encoded_input.items()}
        output = model(**encoded_input)
        x = output.last_hidden_state[:, 0, :].numpy()

    knn = joblib.load(BERT_KNN_PATH)
    pred = knn.predict(x)
    return pred[0]

def BERT(input, tokenizer=None, model=None):
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    if model is None:
        model = BertForSequenceClassification.from_pretrained(BERT_PATH)

    inputs = tokenizer(input, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return predicted_class_id

def GPT(input):
    messages = [
        {
            'role': 'user',
            'content': 'Now I will give you a sentence, you should tell me that it contains positive sentiments or negative sentiments in the first line and tell me the reason.\n Sentence: {}'.format(input)
        }
    ]

    openai.api_key = API_KEY
    openai.api_base = API_BASE
    openai.api_type = API_TYPE
    openai.api_version = API_VERSION

    completion = openai.ChatCompletion.create(
        engine=ENGINE,
        messages=messages
    )

    return completion['choices'][0]['message']['content']