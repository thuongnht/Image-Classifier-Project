from flask import Flask, jsonify, request, Response
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

import configs.config as c
from predict import main


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
SWAGGER_URL = "/api/v1/docs"
API_URL = "/static/swagger/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)


@app.route("/")
@app.route("/api/v1/health-check")
def home():
    return "OK"


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    for h in c.headers:
        print(h['header'], h['value'])
        if request.headers[h['header']] != h['value']:
            return Response(response=f"Request invalid!", status=500)

    model = request.form['model']
    if model is None or model not in c.map_models.keys():
        return Response(response=f"Model {model} invalid. Should be in {c.map_models.keys()}", status=500)

    message = f"User {request.form['model']} received access to server {request.files['file']}"
    return jsonify({
        "Message": message
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
