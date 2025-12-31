from flask import Flask, request, jsonify
import joblib
import re
import string

app = Flask(__name__)

model = None
vectorizer = None

@app.before_first_request
def load_artifacts():
    global model, vectorizer
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("Model and vectorizer loaded.")

# -----------------------------
# Text Preprocessing (IDENTICAL to training)
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.get_json()

    if not data or 'tweet_text' not in data:
        return jsonify({'error': "Missing 'tweet_text' field"}), 400

    cleaned_text = preprocess_text(data['tweet_text'])
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
