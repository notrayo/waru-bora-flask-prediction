import os
# Flask Ngrok extension

# from flask_ngrok import run_with_ngrok
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from flask import Flask
from .config import Config

app = Flask(__name__)
#run_with_ngrok(app)
app.config.from_object(Config)

from .views import views
app.register_blueprint(views)
