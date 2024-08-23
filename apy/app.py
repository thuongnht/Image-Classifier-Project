import json
import os
import time

from flask import Flask, jsonify, request, Response
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from werkzeug.utils import secure_filename
from requests_toolbelt import MultipartEncoder

import configs.config as c
import prediction as p

logger = c.logger(name=__name__)


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
async def predict():
    message, status, content_type, headers = await async_predict_request(req=request)
    res = Response(response=message, status=status, mimetype=content_type)
    for k, v in (headers or {}).items():
        res.headers[k] = v
    return res


async def async_predict_request(req=None):
    for h in c.headers:
        logger.debug("%s -> %s", h['header'], h['value'])
        if req.headers[h['header']] != h['value']:
            return f"Request invalid!", 500, 'text/plain'

    if 'file' not in req.files:
        return f"No file uploaded for prediction!", 500, 'text/plain', None

    file = req.files['file']
    mimetype = file.content_type
    if mimetype not in c.valid_file_mimes:
        return f"file type {mimetype} invalid. Should be in {c.valid_file_mimes}", 500, 'text/plain', None
    logger.debug(f"{file.filename} mime {mimetype}")

    model = req.form['model']
    if model is None or model not in c.map_models.keys():
        return f"Model {model} invalid. Should be in {c.map_models.keys()}", 500, 'text/plain', None
    path_model = os.path.join(c.path_model, c.map_models.get(model))
    logger.debug(path_model)
    if not os.path.exists(path_model):
        return f"Model {model} -> {c.map_models.get(model)} unavailable", 500, 'text/plain', None

    tmp_dir = os.path.join(c.tmp_dir, f"{round(time.time() * 1000)}")
    os.makedirs(tmp_dir, exist_ok=True)
    logger.debug(tmp_dir)

    path_file = os.path.join(tmp_dir, secure_filename(file.filename))
    file.save(path_file)
    logger.debug(path_file)

    topk = int(req.form['topk'])
    logger.debug(topk)

    filename_html = f"prediction_{file.filename.rsplit('.', 1)[0].lower()}.html"
    path_html = os.path.join(tmp_dir, secure_filename(filename_html))
    logger.debug(path_html)

    top_categories, path_html = p.do_predict(path_model=path_model, path_image=path_file, path_html=path_html, topk=topk)
    m = MultipartEncoder(
        fields={
            'top_categories': json.dumps(top_categories),
            'figure': (filename_html, open(path_html, 'rb'), 'text/html')
        }
    )
    return m.to_string(), 200, m.content_type, None  # {'content-disposition': f'attachment; filename="{filename_html}"'}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)

