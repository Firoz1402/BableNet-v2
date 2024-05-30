from flask import Flask, request, jsonify
import pickle
import numpy as np
import re
from modules.simple_ann_for_classification import SimpleANN, sigmoid, relu , softmax
app = Flask(__name__)

# Load the model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data

model_data = load_model('../models/text_classifier.pkl')

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text


# Prediction function
def classify_text(text, model_data):
    text = preprocess_text(text)
    vectorized_text = model_data['tfidf_vectorizer'].transform([text]).toarray()
    
    input_size = vectorized_text.shape[1]
    hidden_size = model_data['weights_input_hidden'].shape[1]
    output_size = model_data['weights_hidden_output'].shape[1]
    
    ann = SimpleANN(input_size, hidden_size, output_size, activation='relu')
    ann.weights_input_hidden = model_data['weights_input_hidden']
    ann.bias_input_hidden = model_data['bias_input_hidden']
    ann.weights_hidden_output = model_data['weights_hidden_output']
    ann.bias_hidden_output = model_data['bias_hidden_output']
    
    prediction = ann.predict(vectorized_text)
    category = model_data['label_encoder'].inverse_transform(prediction)[0]
    
    return category

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    text = data['text']
    predicted_category = classify_text(text, model_data)

    return predicted_category

if __name__ == '__main__':
    app.run(port = 8002, debug=True)
