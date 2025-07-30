# In file: main.py
import json
from engine import UnifiedHeuristicEngine
from models import analyze_text, analyze_visuals, trace_source, verify_with_web
from models.behavioural_profiler import BehaviouralProfiler
from models.audio_analyzer import analyze_audio

def main():
    """
    Main function to run the multi-modal misinformation detection system.
    """
    print("--- Initializing Misinformation Analysis System ---")
    
    # 1. Instantiate the Online Behavioural Profiler
    # It will load 'online_profiles.joblib' if it exists, otherwise start fresh.
    try:
        profiler = BehaviouralProfiler(history_data_path="data/twcs.csv", profile_state_path="assets/online_profiles.joblib")
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        return

    # 2. Define sample input data for a realistic test
    sample_author_id = "AmazonHelp"
    sample_text = "Tom Cruise and Shah Rukh Khan Announce Joint Mission to Colonize Mars on a Private Rocket Built in a Garage."
    sample_timestamp = "2017-11-01T10:30:00Z"
    sample_media_path = "data/sample.mp4"
    sample_audio_path = "data/sample_audio.wav" 

    # 3. Configure and instantiate the engine
    # Define weights for each model's contribution to the final score.
    # The weights must sum to 1.0.
    model_weights = {
        'text': 0.30,
        'visual': 0.15,
        'audio': 0.20,
        'source': 0.20,
        'behavioural': 0.15
    }
    
    # Define the score thresholds for different alert levels.
    alert_thresholds = {'medium': 0.4, 'high': 0.75}

    engine = UnifiedHeuristicEngine(weights=model_weights, thresholds=alert_thresholds)

    print("\n--- Starting Content Analysis ---")
    
    # 4. Run placeholder analysis models to get scores
    # In a real application, you would pass the actual data.
    score_t = analyze_text(sample_text)
    score_v = analyze_visuals(sample_media_path)
    score_s = trace_source(sample_text)
    score_b_1 = profiler.analyze(
        author_id = sample_author_id,
        new_text = sample_text,
        new_timestamp = sample_timestamp
    )
    score_a = analyze_audio(sample_audio_path)
    final_verdict = engine.analyze_content(score_t, score_v, score_s, score_b_1, score_a)

    # 5. Conditionally perform deep web verification for high-risk content
    if final_verdict['alert_level'] in ["High Alert", "Medium Alert"]:
        print("\n--- High-Risk Content Detected: Performing Deep Web Verification ---")
        web_evidence_report = verify_with_web(sample_text)
        final_verdict['web_verification'] = web_evidence_report

    # 6. Print the final, structured result
    print("\n--- VERDICT 1 ---")
    print(json.dumps(final_verdict, indent=2))

    # # --- Test Case 2: Second tweet from 'AmazonHelp' immediately after ---
    print("\n--- [Analysis 2] Starting Content Analysis for a user already in cache ---")
    
    score_b_2 = profiler.analyze(
        author_id="AmazonHelp",
        new_text="Sorry to hear of the trouble. We have responded to your Direct Message. We'll see you there!",
        new_timestamp="2017-11-01T10:35:00Z"
    )

    final_verdict_2 = engine.analyze_content(score_t, score_v, score_s, score_b_2, score_a)
    print("\n--- VERDICT 2 ---")
    print(json.dumps(final_verdict_2, indent=2))

    # 7. Save the updated profiles back to the file for persistence
    profiler.save_profiles()

if __name__ == "__main__":
    main()