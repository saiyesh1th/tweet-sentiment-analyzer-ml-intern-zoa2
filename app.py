from flask import Flask, request, jsonify
import joblib
import re
import string
from nltk.tokenize import word_tokenize
# Make sure to run: nltk.download('punkt') if you haven't already

app = Flask(__name__)

# Placeholder for loaded model and vectorizer
# These should be loaded once when the app starts
model = None
vectorizer = None

@app.before_first_request
def load_artifacts():
    global model, vectorizer
    try:
        model = joblib.load('sentiment_model.pkl') # Update filename if needed
        vectorizer = joblib.load('tfidf_vectorizer.pkl') # Update filename if needed
        print('Model and vectorizer loaded successfully.')
    except FileNotFoundError:
        print('ERROR: Model or vectorizer files not found. Please complete Mission 2 first.')

# Define your text cleaning function here based on Mission 1
def preprocess_text(text):
    # Implement your text cleaning logic here
    # Example: lowercase, remove URLs, mentions, hashtags, punctuation
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    if not request.json or 'tweet_text' not in request.json:
        return jsonify({'error': 'Please provide 'tweet_text' in the request body'}), 400

    tweet_text = request.json['tweet_text']

    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Complete Mission 2 first.'}), 503

    # Preprocess the text
    cleaned_text = preprocess_text(tweet_text)
    # Tokenize (optional, depending on how your vectorizer was trained)
    # tokens = word_tokenize(cleaned_text)
    # Preprocess and vectorize the input text
    text_vectorized = vectorizer.transform([cleaned_text]) # Ensure it's a list

    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    # You might need to map numerical prediction to sentiment string if your model outputs numbers
    sentiment_label = 'positive' if prediction == 1 else 'negative' # Adjust based on your model's output

    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    # For development, set debug=True. For production, use a WSGI server.
    app.run(debug=True, host='0.0.0.0', port=5000)
