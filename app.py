from flask import Flask
from app.routes.inference import inference_api
import nltk


def create_app():
    app = Flask(__name__)

    app.register_blueprint(inference_api, url_prefix="/inference")

    return app


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--port", default=5000, type=int, help="port to listen on")
    args = parser.parse_args()
    port = args.port
    nltk.download("punkt")

    app = create_app()
    app.app_context().push()
    app.run(host="0.0.0.0", port=port)
