import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
model = tf.keras.models.load_model('cnn.keras')

# Ensure uploads folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Route to the home page
@app.route('/')
def home():
    return render_template("index.html")

# Route to the predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")

# Route to the about page
@app.route('/about')
def about():
    return render_template("about.html")

# Route to the contact page
@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route("/output", methods=['GET', 'POST'])
def output():
    if request.method == "POST":
        # Check if the 'image' field exists in the request
        if 'image' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        # Get the file from the request
        f = request.files['image']
        
        # If no file is selected, return an error
        if f.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the uploaded file to the server
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        
        # Load the image and preprocess it
        try:
            img = load_img(filepath, target_size=(224, 224))  # Resize the image to the required size
            image_array = np.array(img) / 255.0  # Convert the image to an array and normalize it
            
            # Add a batch dimension (as the model expects batch inputs)
            image_array = np.expand_dims(image_array, axis=0)
            
            # Use the pre-trained model to make a prediction
            pred_prob = model.predict(image_array)  # Get prediction probabilities
            
            # Get the index of the highest probability
            pred = np.argmax(pred_prob, axis=1)
            
            # Define the possible classes
            index = ['Moon_jellyfish', 'barrel_jellyfish', 'blue_jellyfish', 
                     'compass_jellyfish', 'Lions_mane_jellyfish', 'mauve_sting']
            
            # Get the predicted class
            prediction = index[int(pred)]
            
            # Print the prediction for debugging purposes
            print(f"Prediction: {prediction}")
            
            # Return the result page with the prediction
            return render_template("portfolio-details.html", predict=prediction)
        
        except Exception as e:
            # Handle any exceptions that occur during processing
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
