from flask import Flask, request, jsonify
from modules.english_to_hindi_translator import translate_sentence

app = Flask(__name__)
from flask_cors import CORS
CORS(app)
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    input_sentence = data.get('sentence')
    if not input_sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    translated_sentence = translate_sentence(input_sentence)
    return  translated_sentence

if __name__ == '__main__':
    app.run(port= 8003, debug=True)
