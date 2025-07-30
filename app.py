# app.py (Updated)
from flask import Flask, render_template, request, jsonify
from main import run_analysis
import tempfile
import os
import time

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

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)