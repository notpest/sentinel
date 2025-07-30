# main.py
import json
from engine import UnifiedHeuristicEngine
from models import analyze_text, analyze_visuals, trace_source, verify_with_web
from models.behavioural_profiler import BehaviouralProfiler
from models.audio_analyzer import analyze_audio

def run_analysis(text, media_path="", author_id="AmazonHelp", timestamp="2017-11-01T10:30:00Z"):
    """
    Run multi-modal analysis on the provided input.
    """
    profiler = BehaviouralProfiler(history_data_path="data/twcs.csv", profile_state_path="assets/online_profiles.joblib")

    # Determine media type
    score_v = score_a = 0.0
    if media_path.endswith(".mp4"):
        score_v = analyze_visuals(media_path)
    elif media_path.endswith(".mp3") or media_path.endswith(".wav"):
        score_a = analyze_audio(media_path)

    score_t = analyze_text(text)
    score_s = trace_source(text)
    score_b = profiler.analyze(author_id=author_id, new_text=text, new_timestamp=timestamp)

    model_weights = {
        'text': 0.30,
        'visual': 0.15,
        'audio': 0.20,
        'source': 0.20,
        'behavioural': 0.15
    }
    alert_thresholds = {'medium': 0.4, 'high': 0.75}
    engine = UnifiedHeuristicEngine(weights=model_weights, thresholds=alert_thresholds)

    verdict = engine.analyze_content(score_t, score_v, score_s, score_b, score_a)

    if verdict['alert_level'] in ["High Alert", "Medium Alert"]:
        web_verification = verify_with_web(text)
        verdict["web_verification"] = web_verification

    profiler.save_profiles()
    return verdict
