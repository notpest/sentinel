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

# --- Main route ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("inputText", "")
        uploaded_file = request.files.get("file")
        temp_path = None

        if uploaded_file and uploaded_file.filename:
            suffix = os.path.splitext(uploaded_file.filename)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                uploaded_file.save(temp.name)
                temp_path = temp.name

        result = run_analysis(text=input_text, media_path=temp_path or "")

        if temp_path:
            safe_remove(temp_path)

        return jsonify(result)

    return render_template("index.html")

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)
