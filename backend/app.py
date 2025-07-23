from flask import Flask, request, jsonify
from model import load_model, predict

app = Flask(__name__)

#load model just once when server starts

model = load_model()

@app.route('/predict', methods=['GET'])
def predict_route():
    text = request.args.get('text')
    if not text:
        return jsonify({'error': 'text parameter required'}), 400

    label = predict(text, model)
    return jsonify({'text': text, 'label': label})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
