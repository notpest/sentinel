import json
from engine import UnifiedHeuristicEngine
from models import analyze_text, analyze_visuals, trace_source, verify_with_web
from models.behavioural_profiler import BehaviouralProfiler

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
        return {"error": f"Fatal Error: {str(e)}"}

    # Define model weights (must sum to 1.0)
    model_weights = {
        'text': 0.35,
        'visual': 0.35,
        'source': 0.20,
        'behavioural': 0.10
    }

    # Define alert thresholds
    alert_thresholds = {'medium': 0.4, 'high': 0.75}

    # Initialize engine
    engine = UnifiedHeuristicEngine(weights=model_weights, thresholds=alert_thresholds)

    # Run individual model analyses
    score_t = analyze_text(text) if text else 0
    score_v = analyze_visuals(media_path) if media_path else 0
    score_s = trace_source(text) if text else 0
    score_b = profiler.analyze(
        author_id=author_id,
        new_text=text,
        new_timestamp=timestamp
    ) if text else 0

    # Generate final verdict
    final_verdict = engine.analyze_content(score_t, score_v, score_s, score_b)

    # Optionally verify on web
    if final_verdict['alert_level'] in ["High Alert", "Medium Alert"] and text:
        web_evidence_report = verify_with_web(text)
        final_verdict['web_verification'] = web_evidence_report

    # Persist updated profiles
    profiler.save_profiles()

    return final_verdict
