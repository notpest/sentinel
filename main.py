# main.py
import json
from engine import UnifiedHeuristicEngine
from models import analyze_text, analyze_visuals, trace_source, verify_with_web
from models.behavioural_profiler import BehaviouralProfiler
from models.audio_analyzer import analyze_audio

def _parse_and_adjust_score(verdict, web_report, thresholds):
    """
    Parses the web verification report and adjusts the final score and alert level.
    """
    original_score = verdict['final_score']
    adjusted_score = original_score

    # Adjust score based on web verification verdict
    if "Contradicted" in web_report:
        # If claims are contradicted, significantly increase the score towards 1.0
        # This formula pushes the score 75% of the remaining distance to 1.0
        adjusted_score = original_score + (1.0 - original_score) * 0.75
        print("-> Web verification CONTRADICTED claims. Increasing risk score.")
    elif "Corroborated" in web_report:
        # If claims are corroborated, significantly decrease the score
        # This formula reduces the score by 50%
        adjusted_score = original_score * 0.5
        print("-> Web verification CORROBORATED claims. Decreasing risk score.")
    else:
        print("-> Web verification provided insufficient information. No score adjustment.")

    # Ensure the adjusted score is capped between 0.0 and 1.0
    verdict['final_score'] = min(max(adjusted_score, 0.0), 1.0)
    verdict['original_score'] = round(original_score, 3) # Store original for reference

    # Re-evaluate alert level based on the NEW adjusted score
    if verdict['final_score'] >= thresholds['high']:
        verdict['alert_level'] = "High Alert"
        verdict['headline'] = "Content is highly likely to be malicious or synthetic."
    elif verdict['final_score'] >= thresholds['medium']:
        verdict['alert_level'] = "Medium Alert"
        verdict['headline'] = "Content shows moderate indicators of manipulation. Proceed with caution."
    else:
        verdict['alert_level'] = "Low Alert"
        verdict['headline'] = "Content shows low indicators of manipulation."

    return verdict

def run_analysis(text, media_path="", author_id="AmazonHelp", timestamp="2017-11-01T10:30:00Z"):
    """
    Run multi-modal analysis on the provided input.
    """
    profiler = BehaviouralProfiler(history_data_path="data/twcs.csv", profile_state_path="assets/online_profiles.joblib")

    source_info = trace_source(text)
    score_s = source_info['score']
    predicted_source_model = source_info['model_name']

    score_t = analyze_text(text)
    score_b = profiler.analyze(author_id=author_id, new_text=text, new_timestamp=timestamp)
    web_verification_report = verify_with_web(text)

    score_v = score_a = 0.0
    if media_path.endswith(".mp4"):
        score_v = analyze_visuals(media_path)
    elif media_path.endswith(".mp3") or media_path.endswith(".wav"):
        score_a = analyze_audio(media_path)

    model_weights = {
        'text': 0.30,
        'visual': 0.15,
        'audio': 0.20,
        'source': 0.20,
        'behavioural': 0.15
    }
    alert_thresholds = {'medium': 0.4, 'high': 0.75}
    engine = UnifiedHeuristicEngine(weights=model_weights, thresholds=alert_thresholds)

    initial_verdict = engine.analyze_content(score_t, score_v, score_s, score_b, score_a)
    scores = initial_verdict.pop('component_scores')
    reasoning = [
        f"Textual Disinformation Score: {scores['text']:.2f}",
        f"Visual/Audio Manipulation Score: {max(scores['visual'], scores['audio']):.2f}",
        f"AI Source Tracing Score: {scores['source']:.2f} (Predicted Origin: {predicted_source_model})",
        f"Behavioural Anomaly Score: {scores['behavioural']:.2f}"
    ]
    initial_verdict['reasoning'] = reasoning
    initial_verdict['web_verification_report'] = web_verification_report
    
    print("\n--- [Stage 2] Adjusting Score Based on Web Fact-Checking ---")
    final_verdict = _parse_and_adjust_score(initial_verdict, web_verification_report, alert_thresholds)

    profiler.save_profiles()
    return final_verdict
