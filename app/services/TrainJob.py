from app.services.TrainService import  TrainService
import os
import boto3
import patoolib

ACCESS_ID = os.getenv("ACCESS_ID")
ACCESS_KEY = os.getenv("ACCESS_KEY")

if not os.path.exists("app/resources/sentence_articles_final.csv"):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key=ACCESS_KEY)
    s3.download_file('projet-annuel-5iabd', 'sentence_articles_final.csv', 'app/resources/sentence_articles_final.csv')


service = TrainService('app/resources/sentence_articles_final.csv')
service.fit(10, 2, 0.1)

#os.remove("app/resources/bert_classification.rar")
