from flask import Blueprint, jsonify, make_response, request
from gensim.summarization import summarize

from app.services.BertService import process_bert, tokenize_article_to_sentences, load_model_bert

inference_api = Blueprint("api", __name__)
classifier_bert = load_model_bert()


@inference_api.route("/article", methods=["post"])
def do_bert():
    try:
        data = request.json

        if data["article"]:
            list_sentences = tokenize_article_to_sentences(data["article"])
            sentence_sentiment, positive, negative = process_bert(list_sentences, classifier_bert)

            return make_response(jsonify({
                "data": sentence_sentiment,
                "positives": positive,
                "negatives": negative
            }), 200)
        else:
            return make_response(jsonify(error="article is missing"), 400)
    except:
        return make_response(jsonify(error="error during the process."), 400)


@inference_api.route("/resume", methods=["post"])
def do_resume():
    try:
        data = request.json

        if data["article"]:
            article = data["article"]
            summary = summarize(article)

            return make_response(jsonify({
                "resume": summary
            }), 200)
        else:
            return make_response(jsonify(error="article is missing"), 400)
    except:
        return make_response(jsonify(error="error during the process."), 400)
