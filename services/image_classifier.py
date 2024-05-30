from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle

from modules.simple_ann_for_classification import SimpleANN, sigmoid, relu , softmax
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
# Load the trained model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict the class of a given input image using a loaded model
def predict_image_class(model, image):
    # Preprocess the image (resize, grayscale, normalize)
    resized_image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0  # Normalize to [0, 1]

    # Flatten the image
    flattened_image = normalized_image.reshape(1, -1)

    # Make prediction using the loaded model
    predicted_class_index = model.predict(flattened_image)

    return predicted_class_index[0]

# Load the trained model
loaded_model = load_model('../models/image_classifier_using_mlp.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image file from request
        file = request.files['file']
        
        # Read image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Predict the class of the image
        predicted_class_index = predict_image_class(loaded_model, image)
        label = ''
        if(predicted_class_index==0):
            label = 'Normal Lungs'
        else:
            label = 'Lungs Infected with Pneumonia'
        
        return jsonify({"label": label}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run( port=8001, debug=True)
