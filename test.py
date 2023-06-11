from model import GPT, BERT
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from tqdm import tqdm

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import joblib
import torch
import json

from config import *

def get_acc(pred, label):
    return metrics.accuracy_score(y_true=label, y_pred=pred)

def get_vector(x_val):
    vectorizer = joblib.load(TFIDF_VECTOR_PATH)
    tfidf_val = vectorizer.transform(x_val)

    bert_val = np.load(BERT_VAL_VEC_PATH)

    return tfidf_val, bert_val

def test_gpt(y_val):
    with open(GPT_RESPONCE) as f:
        data = json.load(f)
    pred = []
    for i in data:
        if 'ositive' in i["output"]:
            pred.append(1)
        else:
            pred.append(0)
    
    return get_acc(np.array(pred), y_val)

def test_bert(x_val, y_val):
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_PATH)

    pred = []
    for data in tqdm(x_val):
        output = BERT(data, model=bert_model, tokenizer=tokenizer)
        pred.append(output)
    
    return get_acc(np.array(pred), y_val)

def test_tfidf_lr(tfidf_val, y_val):
    lr = joblib.load(TFIDF_LR_PATH)
    pred = lr.predict(tfidf_val)

    return get_acc(pred, y_val)

def test_bert_lr(bert_val, y_val):
    lr = joblib.load(BERT_LR_PATH)
    pred = lr.predict(bert_val)

    return get_acc(pred, y_val)

def test_tfidf_knn(tfidf_val, y_val):
    knn = joblib.load(TFIDF_KNN_PATH)
    pred = knn.predict(tfidf_val)

    return get_acc(pred, y_val)

def test_bert_knn(bert_val, y_val):
    knn = joblib.load(BERT_KNN_PATH)
    pred = knn.predict(bert_val)

    return get_acc(pred, y_val)

def test(x_val, y_val, type=None):
    tfidf_val, bert_val = get_vector(x_val)
    if type=="Logistic Regression (TFIDF)":
        acc = test_tfidf_lr(tfidf_val, y_val)
    elif type=="Logistic Regression (BERT)":
        acc = test_bert_lr(bert_val, y_val)
    elif type=="KNN (TFIDF)":
        acc = test_tfidf_knn(tfidf_val, y_val)
    elif type=="KNN (BERT)":
        acc = test_bert_knn(bert_val, y_val)
    elif type=="BERT (for classification)":
        acc = test_bert(x_val, y_val)
    elif type=="GPT-3.5-turbo":
        acc = test_gpt(y_val)
    return acc

if __name__ == "__main__":
    data = np.load(DATASET_PATH)
    x_val = data["x_val"]
    y_val = data["y_val"]

    alltype = ["Logistic Regression (TFIDF)", "Logistic Regression (BERT)", "KNN (TFIDF)", "KNN (BERT)", "BERT (for classification)", "GPT-3.5-turbo"]

    for t in alltype:
        print("-------------Starting test {}--------------".format(t))
        acc = test(x_val, y_val, t)
        print("acc: {}".format(acc))