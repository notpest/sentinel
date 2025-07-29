# In file: engine.py
import json

class UnifiedHeuristicEngine:
    """
    A rule-based engine to synthesize outputs from four misinformation models,
    calculate a unified score, and generate a trust-calibrated, context-aware response.
    """
    def __init__(self, weights, thresholds):
        """
        Initializes the engine with specific weights and thresholds for four models.
        """
        # Ensure the number of weights matches the expected models
        if len(weights) != 4:
            raise ValueError("Exactly four weights are required: 'text', 'visual', 'source', 'behavioural'")
        
        # Ensure weights sum to 1.0 for a normalized calculation
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("The sum of all weights must be 1.0")
            
        self.weights = weights
        self.thresholds = thresholds
        print("UnifiedHeuristicEngine initialized for four models.")

    def analyze_content(self, score_text, score_visual, score_source, score_behavioural):
        """
        Analyzes content by taking scores from four models and producing a final verdict.
        """
        scores = {
            'text': min(max(score_text, 0.0), 1.0),
            'visual': min(max(score_visual, 0.0), 1.0),
            'source': min(max(score_source, 0.0), 1.0),
            'behavioural': min(max(score_behavioural, 0.0), 1.0)
        }

        # Weighted calculation using the four scores
        malicious_content_score = (
            self.weights['text'] * scores['text'] +
            self.weights['visual'] * scores['visual'] +
            self.weights['source'] * scores['source'] +
            self.weights['behavioural'] * scores['behavioural']
        )

        # Tiered response generation based on score
        if malicious_content_score >= self.thresholds['high']:
            alert_level = "High Alert"
            headline = "Content is highly likely to be malicious or synthetic."
        elif malicious_content_score >= self.thresholds['medium']:
            alert_level = "Medium Alert"
            headline = "Content shows moderate indicators of manipulation. Proceed with caution."
        else:
            alert_level = "Low Alert"
            headline = "Content shows low indicators of manipulation."

        # Build the explanation based on which models had high scores
        reasoning = []
        if scores['text'] > 0.5:
            reasoning.append(f"Textual analysis flagged potential disinformation (Score: {scores['text']:.2f})")
        if scores['visual'] > 0.5:
            reasoning.append(f"Visual analysis flagged potential deepfake artifacts (Score: {scores['visual']:.2f})")
        if scores['source'] > 0.5:
            reasoning.append(f"Source tracing flagged AI-generated text origin (Score: {scores['source']:.2f})")
        # if scores['behavioural'] > 0.5:
        #     reasoning.append(f"Behavioural profiling flagged anomalous patterns (Score: {scores['behavioural']:.2f})")

        reasoning.append(f"Behavioural Anomaly Score: {scores['behavioural']:.2f} (Higher score means more anomalous)")
        
        # This check is now less likely to be triggered, but remains as a fallback.
        if not reasoning:
            reasoning.append("No single strong indicator was found; the score is an aggregate of multiple weak signals.")

        # Assemble and return the final response object
        response = {
            'final_score': round(malicious_content_score, 3),
            'alert_level': alert_level,
            'headline': headline,
            'reasoning': reasoning
        }
        return response