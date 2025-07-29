import nltk

# This script downloads the necessary data packages for our project.
# We need 'stopwords', 'punkt' for tokenization, and 'wordnet' for lemmatization.
print("Starting NLTK data download...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')
print("✅ NLTK data download complete.")