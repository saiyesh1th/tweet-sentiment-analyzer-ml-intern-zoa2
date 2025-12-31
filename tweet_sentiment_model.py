import pandas as pd
import re
import string
import joblib

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/tweets.csv")

# Sanity check
assert {'text', 'sentiment'}.issubset(df.columns), "CSV must contain text and sentiment columns"

# -----------------------------
# 2. Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['cleaned_text'] = df['text'].apply(clean_text)

# -----------------------------
# 3. Tokenization (explicit)
# -----------------------------
df['tokens'] = df['cleaned_text'].apply(word_tokenize)

# Join tokens back (TF-IDF expects strings)
df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))

# -----------------------------
# 4. Encode Labels
# -----------------------------
df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# -----------------------------
# 5. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['processed_text'])
y = df['label']

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 7. Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# -----------------------------
# 9. Save Artifacts
# -----------------------------
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully.")
