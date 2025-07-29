import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Efficiently Load Model and Vectorizer Once ---

# Define the path to the assets directory relative to this file's location
ASSETS_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets')
MODEL_PATH = os.path.join(ASSETS_PATH, 'fake_news_model.joblib')
VECTORIZER_PATH = os.path.join(ASSETS_PATH, 'tfidf_vectorizer.joblib')

try:
    # Load the trained model and vectorizer from the assets folder
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Textual analysis model and vectorizer loaded successfully.")
    
    # Initialize NLTK components for preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

except FileNotFoundError:
    print("❌ Error: Model or vectorizer file not found. Make sure the 'assets' directory is correctly placed.")
    model = None
    vectorizer = None

# --- Preprocessing Function from our Notebook ---

def preprocess_text(text):
    """Cleans and preprocesses a text string."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)

# --- Main Analysis Function ---

def analyze_text(text_content):
    """
    Analyzes text content using the loaded TF-IDF and Logistic Regression model.
    Returns the probability of the text being FAKE (malicious).
    """
    print("-> Running real-time textual analysis...")
    
    if not model or not vectorizer:
        print("   Model not loaded. Returning a neutral score.")
        return 0.5 # Return a neutral score if the model failed to load

    # 1. Preprocess the input text
    preprocessed_text = preprocess_text(text_content)
    
    # 2. Transform the text using the loaded vectorizer
    vectorized_text = vectorizer.transform([preprocessed_text])
    
    # 3. Get the prediction probabilities from the model
    # The output is in the format: [[prob_class_0, prob_class_1]]
    probabilities = model.predict_proba(vectorized_text)
    
    # 4. Extract and return the probability for the "FAKE" class (index 1)
    # This score represents the likelihood of the content being malicious.
    fake_news_score = probabilities[0][1]
    
    return fake_news_score