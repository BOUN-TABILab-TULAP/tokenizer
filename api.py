from flask import Flask, json, g, request, jsonify, json
import bap_preprocessing
app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    json_data = json.loads(request.data)
    response=bap_preprocessing.tokenize(json_data['text'])

    result = {"tokens": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False,)