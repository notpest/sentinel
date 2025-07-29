import nltk

def download_nltk_resources():
    """
    Downloads the necessary NLTK data packages for the project.
    """
    print("--- Downloading NLTK Resources ---")
    try:
        # 'punkt' is for tokenization
        nltk.data.find('tokenizers/punkt')
        print("✅ 'punkt' is already downloaded.")
    except LookupError:
        print("   Downloading 'punkt'...")
        nltk.download('punkt', quiet=True)
        print("✅ 'punkt' downloaded successfully.")

    try:
        # 'wordnet' is for lemmatization
        nltk.data.find('corpora/wordnet')
        print("✅ 'wordnet' is already downloaded.")
    except LookupError:
        print("   Downloading 'wordnet'...")
        nltk.download('wordnet', quiet=True)
        print("✅ 'wordnet' downloaded successfully.")

    print("--- All necessary resources are ready. ---")

if __name__ == "__main__":
    download_nltk_resources()