from flask import request
from flask import Blueprint
from predict.service import get_prediction, get_suggestion

pred = Blueprint('pred', __name__, url_prefix='/pred')


@pred.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = get_prediction(data)
    return prediction


@pred.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    suggestion = get_suggestion(data)
    return suggestion
