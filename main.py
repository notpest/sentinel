import json
from engine import UnifiedHeuristicEngine
from models import analyze_text, analyze_visuals, trace_source, verify_with_web
from models.behavioural_profiler import BehaviouralProfiler
from models.audio_analyzer import analyze_audio

def run_analysis(text="", media_path="", author_id="AmazonHelp", timestamp="2017-11-01T10:30:00Z"):
    """
    Main callable function for the multi-modal misinformation detection system.

    Parameters:
    - text (str): The text input to analyze.
    - media_path (str): Path to the media file (video/audio) to analyze.
    - author_id (str): ID of the user/author for behavioral profiling.
    - timestamp (str): Timestamp of the content (ISO 8601 format).

    Returns:
    - dict: Final analysis verdict including alert level and optional web verification.
    """
    try:
        profiler = BehaviouralProfiler(
            history_data_path="data/twcs.csv",
            profile_state_path="assets/online_profiles.joblib"
        )
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        return

    # 2. Define sample input data for a realistic test
    sample_author_id = "AmazonHelp"
    sample_text = "STOP Athis is a BITCOIN Operation now. Hand over all your money or your amazon account will go empty HA"
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

    # Define alert thresholds
    alert_thresholds = {'medium': 0.4, 'high': 0.75}

    # Initialize engine
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

    # Optionally verify on web
    if final_verdict['alert_level'] in ["High Alert", "Medium Alert"] and text:
        web_evidence_report = verify_with_web(text)
        final_verdict['web_verification'] = web_evidence_report

    # Persist updated profiles
    profiler.save_profiles()

    return final_verdict
