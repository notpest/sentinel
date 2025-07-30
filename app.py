# app.py (Updated)
from flask import Flask, render_template, request, jsonify
from main import run_analysis
from models.behavioural_profiler import BehaviouralProfiler
import tempfile
import os
import time
import pandas as pd

app = Flask(__name__)

# --- Safely remove file with retry ---
def safe_remove(path, retries=10, delay=0.5):
    for attempt in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError:
            time.sleep(delay)
    print(f"❌ Failed to delete temp file after {retries} retries: {path}")

# --- Route to render the main page ---
@app.route("/")
def index():
    """Renders the main HTML page."""
    return render_template("index.html")

# --- API endpoint for running the analysis ---
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives data, runs the analysis, and returns a JSON result.
    This is the endpoint our JavaScript will call.
    """
    input_text = request.form.get("inputText", "")
    uploaded_file = request.files.get("file")
    temp_path = None

    if uploaded_file and uploaded_file.filename:
        suffix = os.path.splitext(uploaded_file.filename)[1] or ".tmp"
        # Use a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            uploaded_file.save(temp.name)
            temp_path = temp.name

    # Run the time-consuming analysis
    result = run_analysis(text=input_text, media_path=temp_path or "")

    # Clean up the temporary file after analysis
    if temp_path:
        safe_remove(temp_path)

    return jsonify(result)

@app.route("/profiler")
def profiler_test_page():
    """Renders the dedicated test page for the behavioural profiler."""
    return render_template("profiler_test.html")

@app.route("/test_profiler", methods=['POST'])
def test_profiler_endpoint():
    """
    API endpoint to run only the behavioural profiler and return detailed results.
    """
    data = request.get_json()
    author_id = data.get('authorId')
    text = data.get('text')
    timestamp = data.get('timestamp', pd.Timestamp.now().isoformat())

    if not author_id or not text:
        return jsonify({"error": "authorId and text are required"}), 400

    # Instantiate the profiler for this request
    profiler = BehaviouralProfiler(history_data_path="data/twcs.csv")
    
    # Call the new detailed method
    detailed_result = profiler.analyze_and_explain(author_id, text, timestamp)
    
    # Save the updated profile state
    profiler.save_profiles()
    
    return jsonify(detailed_result)

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)