from models import GPT, BERT
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from tqdm import tqdm


def test_BERT(val, bert_path):
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = BertForSequenceClassification.from_pretrained(bert_path)

    correct, total = 0, 0
    for data in tqdm(val):
        input = data['sentence']
        label = data['label']
        output = BERT(bert_model, tokenizer, input)

        if all((label==1, 'Positive' in output)):
            correct += 1
        elif all((label==0, 'Negative' in output)):
            correct += 1
        else:
            print(input)
        total += 1
    return correct / total



if __name__ == "__main__":
    dataset = load_dataset("sst2")
    val = dataset['validation']
    
    path = "/Users/mccartney/downloads/bert-base-uncased-SST-2"

    acc = test_BERT(val, path)

    print(acc)