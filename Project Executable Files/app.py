import os
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, abort
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import tempfile

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB limit

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configure logging
logging.basicConfig(level=logging.ERROR, filename='app.log', format='%(asctime)s %(levelname)s:%(message)s')

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
try:
    model = load_model('cnn.keras')
except Exception as e:
    logging.error("Error loading the model", exc_info=True)
    raise RuntimeError("Failed to load the model. Check the path and configuration.")

# Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/output", methods=['POST'])
def output():
    # File upload logic...
    pass  # The rest of your code goes here

if __name__ == '__main__':
    app.run(debug=False, port=8080)
