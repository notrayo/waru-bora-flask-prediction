from flask import Blueprint, request, jsonify, url_for, send_file
from flask import render_template
from werkzeug.utils import secure_filename
from . import app
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from . import app

# JSONIFY

from flask import jsonify

plt.switch_backend('agg')
views = Blueprint("views", __name__)
@views.route('/', methods=['GET','POST'])
def home():
    if request.method == "POST":
        file = request.files['file']
        input_img = secure_filename(file.filename)
        file.save(app.config['IMAGE_UPLOADS']+input_img)

        pred=predict_save(input_img)
        # image for rendering on Flutter
        pred_img_filename = 'pred_img.png'
        predict_img_path = app.config['IMAGE_UPLOADS'] + pred_img_filename
        pred_img_url = request.host_url + url_for('static', filename='uploads/' + pred_img_filename)

        return jsonify({'prediction': pred, 'prediction_image_url': pred_img_url})
    return render_template('home.html')


@views.route('/about')
def about():
    return render_template('about.html')

##############################################
model = load_model(app.config['MODEL'])
class_names = ['Early_blight', 'Healthy', 'Late_blight']

def predict_save(img):
    my_image = load_img(app.config['IMAGE_UPLOADS']+img, target_size=(128, 128))
    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, 0)
    
    out = np.round(model.predict(my_image)[0], 2)
    fig = plt.figure(figsize=(8, 5))
    plt.barh(class_names, 
            [1,1,1], 
            edgecolor='gray',
            linewidth=2,
            color='white',
            height=0.5)
    plt.barh(class_names,
             out, 
             color='lightgray', 
             height=0.5)
    
    for index, value in enumerate(out):
        plt.text(value/2, index, f"{100*value:.2f}%",fontsize=13, fontweight='bold')
        
    plt.xticks([])
    plt.yticks([0, 1, 2], labels=class_names, fontweight='bold', fontsize=14)
    name = app.config['IMAGE_UPLOADS']+'pred_img.png'
    fig.savefig(name, bbox_inches='tight')
    
    return class_names[np.argmax(out)]