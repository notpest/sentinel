import joblib
import librosa
import numpy as np
import os

# --- 1. Load the Pre-Trained Model ---
# Construct the path to the model file relative to this script's location.
# This makes the script more portable.
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "..", "assets", "fake_audio_detector.pkl")

# Load the model once when the module is imported to avoid reloading on every call.
try:
    MODEL = joblib.load(MODEL_PATH)
    print("✅ Audio deepfake model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    MODEL = None

# --- 2. Implement the Feature Extraction Function ---
# This function must be identical to the one used to train the model.
def extract_features(file_path, n_mfcc=13):
    """Extracts MFCC features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None

# --- 3. Implement the Main Analysis Function ---
def analyze_audio(media_path):
    """
    Analyzes an audio file to detect deepfake characteristics.

    Note: The function name is 'analyze_visuals' to match the engine's interface,
    but it processes audio data.
    
    Args:
        media_path (str): The file path to the audio file (.wav, .mp3, etc.).

    Returns:
        float: A score between 0.0 and 1.0, where a higher score indicates a
               higher probability of the audio being a deepfake.
    """
    print("-> Running audio deepfake analysis...")

    if MODEL is None:
        print("Model is not loaded. Returning neutral score.")
        return 0.5 # Return a neutral score if the model failed to load.
        
    # Step 1: Extract features from the new audio file.
    features = extract_features(media_path)
    if features is None:
        return 0.0 # Return a low score if feature extraction fails.

    # Step 2: Reshape features for the model.
    # The model expects a 2D array: (n_samples, n_features).
    # We are predicting on 1 sample.
    reshaped_features = features.reshape(1, -1)

    # Step 3: Get the prediction probability.
    # predict_proba() returns probabilities for each class: [[prob_real, prob_fake]]
    prediction_probabilities = MODEL.predict_proba(reshaped_features)
    
    # The score is the probability of the "fake" class (label 1).
    malicious_score = prediction_probabilities[0][1]
    
    return malicious_score