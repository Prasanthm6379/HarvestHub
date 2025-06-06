from flask import Flask
from flask_cors import CORS
from predict.view import pred


def create_app():
    app = Flask(__name__)
    CORS(app, origins="*")
    app.register_blueprint(pred)

    return app
