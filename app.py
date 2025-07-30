from flask import Flask, render_template, request, jsonify
from main import run_analysis
import tempfile, os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("inputText", "")
        file_type = request.form.get("fileType", "")
        uploaded_file = request.files.get("file")

        temp_path = None
        if uploaded_file:
            suffix = ".mp3" if file_type == "audio" else ".mp4"
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            uploaded_file.save(temp.name)
            temp_path = temp.name

        result = run_analysis(text=input_text, media_path=temp_path or "")

        if temp_path:
            os.remove(temp_path)

        return jsonify(result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
