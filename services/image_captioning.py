from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

app = Flask(__name__)
from flask_cors import CORS
CORS(app)
# Load the model and processor
model = VisionEncoderDecoderModel.from_pretrained("../models/image_captioning/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("../models/image_captioning/vit-image-processor")
tokenizer = AutoTokenizer.from_pretrained("../models/image_captioning/vit-gpt2-tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = Image.open(request.files['image']).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    output_ids = model.generate(pixel_values)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(port=8004)
