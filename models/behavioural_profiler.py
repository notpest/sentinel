# All initial imports and the class definition up to the analyze method remain the same.
# ... (imports, __init__, save_profiles, _extract_stylometric_features, etc.) ...
import pandas as pd
import numpy as np
import nltk
import re
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import os

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    print("-> Downloading NLTK data for behavioural profiler...")
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

class BehaviouralProfiler:
    def __init__(self, history_data_path, profile_state_path='assets/online_profiles.joblib'):
        print("-> Initializing Online Behavioural Profiler...")
        self.history_data_path = history_data_path
        self.profile_state_path = profile_state_path
        self.sid = SentimentIntensityAnalyzer()
        if os.path.exists(self.profile_state_path):
            print(f"-> Loading existing profile state from {self.profile_state_path}")
            self.profile_cache = joblib.load(self.profile_state_path)
        else:
            print("-> No existing profile state found. Starting with an empty cache.")
            self.profile_cache = {}
        
        # Get feature names from the extractor method to use as labels
        self.stylometric_feature_names = list(self._extract_stylometric_features("").keys())
        self.num_stylometric_features = len(self.stylometric_feature_names)
        
        print("-> Online Behavioural Profiler initialized successfully.")

    def save_profiles(self):
        print(f"-> Saving profile state to {self.profile_state_path}...")
        os.makedirs(os.path.dirname(self.profile_state_path), exist_ok=True)
        joblib.dump(self.profile_cache, self.profile_state_path)
        print("-> Save complete.")
    
    def _extract_stylometric_features(self, text):
        if not isinstance(text, str): text = ""
        words_original_case = word_tokenize(text)
        words_lower_case = [w.lower() for w in words_original_case]
        if not words_lower_case:
            return {'text_length': 0, 'word_count': 0, 'avg_word_length': 0, 'ttr': 0, 
                    'mention_count': 0, 'hashtag_count': 0, 'sentiment': 0.0,
                    'uppercase_ratio': 0.0, 'exclamation_count': 0, 'question_count': 0}
        text_length = len(text)
        word_count = len(words_lower_case)
        avg_word_length = np.mean([len(w) for w in words_lower_case]) if word_count > 0 else 0
        ttr = len(set(words_lower_case)) / word_count if word_count > 0 else 0
        mention_count = len(re.findall(r'@\w+', text))
        hashtag_count = len(re.findall(r'#\w+', text))
        sentiment = self.sid.polarity_scores(text)['compound']
        uppercase_words = [w for w in words_original_case if w.isupper() and len(w) > 1]
        uppercase_ratio = len(uppercase_words) / word_count if word_count > 0 else 0
        exclamation_count = text.count('!')
        question_count = text.count('?')
        return {'text_length': text_length, 'word_count': word_count, 'avg_word_length': avg_word_length, 'ttr': ttr, 
                'mention_count': mention_count, 'hashtag_count': hashtag_count, 'sentiment': sentiment,
                'uppercase_ratio': uppercase_ratio, 'exclamation_count': exclamation_count, 'question_count': question_count}

    def _create_empty_profile(self):
        return {
            'feature_sums': np.zeros(self.num_stylometric_features),
            'hourly_counts': np.zeros(24),
            'total_tweets': 0
        }

    def _update_profile_with_tweet(self, profile, text, timestamp):
        features = self._extract_stylometric_features(text)
        feature_vector = np.array(list(features.values()))
        profile['feature_sums'] += feature_vector
        hour = pd.to_datetime(timestamp, errors='coerce').hour
        if pd.notna(hour):
            profile['hourly_counts'][hour] += 1
        profile['total_tweets'] += 1

    def _build_profile_from_history(self, author_id):
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
        if profile['total_tweets'] == 0:
            return None
        mean_stylometric_vector = profile['feature_sums'] / profile['total_tweets']
        mean_temporal_vector = profile['hourly_counts'] / profile['total_tweets']
        return np.concatenate([mean_stylometric_vector, mean_temporal_vector])

    # --- NEW: Detailed analysis method for testing ---
    def analyze_and_explain(self, author_id, new_text, new_timestamp):
        """
        Performs a full analysis and returns a detailed dictionary for inspection.
        """
        print(f"-> Running DETAILED behavioural profiling for '{author_id}'...")
        
        if author_id not in self.profile_cache:
            profile = self._build_profile_from_history(author_id)
            if profile is None: return {"error": "History file not found."}
            self.profile_cache[author_id] = profile
        else:
            profile = self.profile_cache[author_id]

        live_profile_vector = self._get_live_profile_vector(profile)
        
        new_stylometric_features = self._extract_stylometric_features(new_text)
        new_stylometric_vector = np.array(list(new_stylometric_features.values()))
        new_temporal_vector = np.zeros(24)
        hour = pd.to_datetime(new_timestamp, errors='coerce').hour
        if pd.notna(hour): new_temporal_vector[hour] = 1
        new_content_vector = np.concatenate([new_stylometric_vector, new_temporal_vector])
        
        explanation = {}
        if live_profile_vector is None:
            anomaly_score = 0.5
            explanation['summary'] = "User profile is new or empty. Anomaly score is neutral."
        else:
            anomaly_score = cosine(new_content_vector, live_profile_vector)
            explanation['summary'] = (
                f"Comparison of new content against a historical baseline of {profile['total_tweets']} tweets. "
                f"Score indicates cosine distance (higher is more anomalous)."
            )
            # Combine feature names with their values for a readable profile
            explanation['historical_average_profile'] = {
                'stylometric': dict(zip(self.stylometric_feature_names, live_profile_vector[:self.num_stylometric_features].round(4))),
                'temporal': list(live_profile_vector[self.num_stylometric_features:].round(4))
            }
            explanation['new_content_profile'] = {
                'stylometric': dict(zip(self.stylometric_feature_names, new_content_vector[:self.num_stylometric_features].round(4))),
                'temporal': list(new_content_vector[self.num_stylometric_features:].round(4))
            }
        
        self._update_profile_with_tweet(profile, new_text, new_timestamp)
        
        return {
            'author_id': author_id,
            'anomaly_score': min(max(anomaly_score, 0.0), 1.0),
            'explanation': explanation
        }

    # --- ORIGINAL: Simplified method for the main pipeline ---
    def analyze(self, author_id, new_text, new_timestamp):
        """
        Public method for the main pipeline. Returns only the anomaly score.
        This ensures backward compatibility with main.py.
        """
        # This method now calls the detailed one and extracts just the score
        result = self.analyze_and_explain(author_id, new_text, new_timestamp)
        return result.get('anomaly_score', 0.5) # Return 0.5 if there was an error