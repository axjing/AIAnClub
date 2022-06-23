# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model
"""

import argparse
import io
import os
import torch
from flask import Flask, request,render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"
@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        desc = request.form.get('desc')  # 获取描述信息
        avatar = request.files.get('avatar')  # 获取文件：request.files
        filename = secure_filename(avatar.filename)  # 防黑客在文件名上做手脚：../../User/xxx/.bashrc
        avatar.save(os.path.join('files', filename))  # 保存文件
        print(desc)
        return '文件上传成功'
    return render_template('upload.html')

@app.route(DETECTION_URL, methods=["GET"])
def predict():
    if request.method != "GET":
        return

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        results = model(im, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
