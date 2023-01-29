from flask import Flask,request, render_template, session
from infer_on_single_image import getModel, inference_on_single_labelled_image_pca_web_original
import os
import base64
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/temp'
app.secret_key = os.urandom(24)

model_oxford = getModel(weights_file="./static/weights/oxbuild_final.pth")
model_paris = getModel(weights_file="./static/weights/paris_final.pth")

# Main page
@app.route("/")
def index():
    similar_images = []
    filename = ''
    if 'similar_images' in session:
        similar_images = session['similar_images']
    if 'filename' in session:
        filename = session['filename']
    return render_template("main.html", 
                           filename=filename, 
                           evaluated=similar_images,
                           gt = [0]*60)

@app.route("/", methods=['GET', 'POST'])
def imageRetrieval():
    
    if request.method == 'POST' and 'blobimg' in request.form:
        img = request.form.get("blobimg")
        img = img.replace("data:image/jpeg;base64,","")
        imgdata = base64.b64decode(str(img))
        filename = f'./static/temp/temp.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)
        if request.form.get('imgsets') == "oxbuild":
            similar_images, similarity_scores = inference_on_single_labelled_image_pca_web_original(
                model_oxford,
                filename,
                img_dir="./static/data/oxbuild/images/",
                img_fts_dir="./static/fts_pca/oxbuild/",)
        else:
            similar_images, similarity_scores = inference_on_single_labelled_image_pca_web_original(
                model_paris,
                filename,
                img_dir="./static/data/paris/images/",
                img_fts_dir="./static/fts_pca/paris/",)
        session['similar_images'] = similar_images
        session['filename'] = filename
        return {
                "evaluated":similar_images,
                "score": similarity_scores.tolist()
                }
    return 'Something wrong'

app.run(host='0.0.0.0', port=8888)