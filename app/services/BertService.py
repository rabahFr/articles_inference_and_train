import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import boto3
import patoolib

ACCESS_ID = os.getenv("ACCESS_ID")
ACCESS_KEY = os.getenv("ACCESS_KEY")

PATH_TO_MODEL = "app/resources/bert_classification"
BUCKET = "projet-annuel-5iabd"
FILE_NAME = "bert_classification.rar"


def extract_model():
    if not os.path.exists(PATH_TO_MODEL):
        s3 = boto3.client("s3", aws_access_key_id=ACCESS_ID, aws_secret_access_key=ACCESS_KEY)
        s3.download_file(BUCKET, FILE_NAME, PATH_TO_MODEL)
        if os.path.exists("app/resources/bert_classification.rar"):
            patoolib.extract_archive("app/resources/bert_classification.rar", outdir="app/resources/")
            os.remove("app/resources/bert_classification.rar")


def load_model_bert():
    extract_model()
    return AutoModelForSequenceClassification.from_pretrained(PATH_TO_MODEL)


def load_tokenizer_bert():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


def classify_bert(sentence: str, tokenizer, model):
    sentence_tokenized = tokenizer(sentence, return_tensors="pt")
    sentence_output = model(**sentence_tokenized)

    return F.softmax(sentence_output[0], dim=-1)


def map_classification_bert(prediction_output):
    if prediction_output[0][0] > prediction_output[0][1]:
        return "Negative"
    else:
        return "Positive"


def tokenize_article_to_sentences(article):
    return sent_tokenize(article)


def process_bert(list_sentence, model):
    tokenizer = load_tokenizer_bert()
    negative = 0
    positive = 0

    predictions_to_return = []

    for sentence in list_sentence:
        pred = classify_bert(sentence, tokenizer, model)
        state = map_classification_bert(pred)

        prediction_to_return = {"sentence": sentence, "sentiment": state}

        predictions_to_return.append(prediction_to_return)

        if state == "Negative":
            negative += 1
        if state == "Positive":
            positive += 1

    return predictions_to_return, positive, negative
