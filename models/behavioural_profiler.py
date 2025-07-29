# In file: models/behavioural_profiler.py

import pandas as pd
import numpy as np
import nltk
import re
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import os

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    print("-> Downloading NLTK data for behavioural profiler...")
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

class BehaviouralProfiler:
    """
    Analyzes user behaviour using incrementally updated online profiles.
    """
    def __init__(self, history_data_path, profile_state_path='assets/online_profiles.joblib'):
        print("-> Initializing Online Behavioural Profiler...")
        self.history_data_path = history_data_path
        self.profile_state_path = profile_state_path
        self.sid = SentimentIntensityAnalyzer()

        # Load existing profiles from state file, or start fresh
        if os.path.exists(self.profile_state_path):
            print(f"-> Loading existing profile state from {self.profile_state_path}")
            self.profile_cache = joblib.load(self.profile_state_path)
        else:
            print("-> No existing profile state found. Starting with an empty cache.")
            self.profile_cache = {}
        
        # Determine the number of features for creating empty profiles
        self.num_stylometric_features = len(self._extract_stylometric_features(""))
        
        print("-> Online Behavioural Profiler initialized successfully.")

    def save_profiles(self):
        """Saves the current profile cache to a file."""
        print(f"-> Saving profile state to {self.profile_state_path}...")
        os.makedirs(os.path.dirname(self.profile_state_path), exist_ok=True)
        joblib.dump(self.profile_cache, self.profile_state_path)
        print("-> Save complete.")

    def _extract_stylometric_features(self, text):
        # This is our enhanced feature extractor from before. No changes needed.
        # ... (Insert the full _extract_stylometric_features method from the previous response here) ...
        if not isinstance(text, str): text = ""
        
        # We need the original case for uppercase analysis, and lowercase for token analysis
        words_original_case = word_tokenize(text)
        words_lower_case = [w.lower() for w in words_original_case]

        if not words_lower_case:
            return {'text_length': 0, 'word_count': 0, 'avg_word_length': 0, 'ttr': 0, 
                    'mention_count': 0, 'hashtag_count': 0, 'sentiment': 0.0,
                    'uppercase_ratio': 0.0, 'exclamation_count': 0, 'question_count': 0}

        # --- Existing Features ---
        text_length = len(text)
        word_count = len(words_lower_case)
        avg_word_length = np.mean([len(w) for w in words_lower_case]) if word_count > 0 else 0
        ttr = len(set(words_lower_case)) / word_count if word_count > 0 else 0
        mention_count = len(re.findall(r'@\w+', text))
        hashtag_count = len(re.findall(r'#\w+', text))
        sentiment = self.sid.polarity_scores(text)['compound']

        # --- NEW, more sensitive features ---
        # Ratio of uppercase words (e.g., "LOL", "TRASH")
        uppercase_words = [w for w in words_original_case if w.isupper() and len(w) > 1]
        uppercase_ratio = len(uppercase_words) / word_count if word_count > 0 else 0
        
        # Punctuation for tonal analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')

        return {'text_length': text_length, 'word_count': word_count, 'avg_word_length': avg_word_length, 'ttr': ttr, 
                'mention_count': mention_count, 'hashtag_count': hashtag_count, 'sentiment': sentiment,
                'uppercase_ratio': uppercase_ratio, 'exclamation_count': exclamation_count, 'question_count': question_count}

    def _create_empty_profile(self):
        """Creates a blank profile structure with all aggregates set to zero."""
        return {
            'feature_sums': np.zeros(self.num_stylometric_features),
            'hourly_counts': np.zeros(24),
            'total_tweets': 0
        }

    def _update_profile_with_tweet(self, profile, text, timestamp):
        """Updates a profile's aggregates with data from a single new tweet."""
        features = self._extract_stylometric_features(text)
        feature_vector = np.array(list(features.values()))
        
        profile['feature_sums'] += feature_vector
        hour = pd.to_datetime(timestamp, errors='coerce').hour
        if pd.notna(hour):
            profile['hourly_counts'][hour] += 1
        profile['total_tweets'] += 1

    def _build_profile_from_history(self, author_id):
        """Builds a user's initial aggregate profile from the historical CSV."""
        print(f"-> Building initial profile for '{author_id}' from history...")
        try:
            history_df = pd.read_csv(self.history_data_path)
            user_history = history_df[(history_df['author_id'] == author_id) & (history_df['inbound'] == False)]
        except FileNotFoundError:
            print(f"-> FATAL: Could not find history file at {self.history_data_path}")
            return None

        if user_history.empty:
            print(f"-> WARNING: No history found for '{author_id}'. Starting with empty profile.")
            return self._create_empty_profile()

        new_profile = self._create_empty_profile()
        for _, row in user_history.iterrows():
            self._update_profile_with_tweet(new_profile, row['text'], row['created_at'])
        
        print(f"-> Profile built for '{author_id}' from {len(user_history)} historical tweets.")
        return new_profile

    def _get_live_profile_vector(self, profile):
        """Calculates the current mean vector from a profile's aggregates."""
        if profile['total_tweets'] == 0:
            return None # Cannot calculate mean for empty profile

        mean_stylometric_vector = profile['feature_sums'] / profile['total_tweets']
        mean_temporal_vector = profile['hourly_counts'] / profile['total_tweets']
        
        return np.concatenate([mean_stylometric_vector, mean_temporal_vector])

    def analyze(self, author_id, new_text, new_timestamp):
        """
        Public method to analyze new content, return an anomaly score, and update the profile.
        """
        print(f"-> Running online behavioural profiling for '{author_id}'...")
        
        # 1. Get or create the user's profile object
        if author_id not in self.profile_cache:
            profile = self._build_profile_from_history(author_id)
            if profile is None: return 0.5 # Return neutral if history file is missing
            self.profile_cache[author_id] = profile
        else:
            profile = self.profile_cache[author_id]

        # 2. Get the profile's current state vector for comparison
        live_profile_vector = self._get_live_profile_vector(profile)
        if live_profile_vector is None:
            print("-> Warning: Profile is empty. Returning neutral score.")
            anomaly_score = 0.5
        else:
            # 3. Vectorize the new content
            new_stylometric_vector = np.array(list(self._extract_stylometric_features(new_text).values()))
            new_temporal_vector = np.zeros(24)
            hour = pd.to_datetime(new_timestamp, errors='coerce').hour
            if pd.notna(hour):
                new_temporal_vector[hour] = 1
            new_content_vector = np.concatenate([new_stylometric_vector, new_temporal_vector])
            
            # 4. Calculate anomaly score
            anomaly_score = cosine(new_content_vector, live_profile_vector)

        # 5. CRITICAL: Update the profile with the new tweet for the next analysis
        self._update_profile_with_tweet(profile, new_text, new_timestamp)
        print(f"-> Profile for '{author_id}' updated. Total tweets in profile: {profile['total_tweets']}")
        
        # Ensure score is between 0 and 1
        return min(max(anomaly_score, 0.0), 1.0)