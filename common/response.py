from flask import jsonify
from common.strings import Strings


def success_response(statuscode=200, content="Success"):
    responseData = {
        'status': Strings.success,
        'status_code': statuscode,
        'content': content
    }
    resp = jsonify(responseData)
    resp.headers.add('Access-Control-Allow-Credentials', 'true')
    return resp, statuscode


def failure_response(statuscode=500, content="Internal Server Error"):
    responseData = {
        'status': Strings.failure,
        'status_code': statuscode,
        'content': content,
    }
    return jsonify(responseData), statuscode
