# Deepfake Detection System 🎭🔍

This project implements a **deepfake detection pipeline** using pre-trained models and ensemble inference. It processes videos, extracts faces, applies augmentations, and predicts whether a video is **REAL** or **FAKE**.  

## 📌 Features
- Pretrained deepfake detection models.
- Ensemble inference for improved accuracy.
- Face detection and preprocessing with `dlib` and `opencv`.
- Configurable inference using `.sh` scripts.
- Docker support for containerized deployment.

---

## 📂 Project Structure
📦 dfdc_deepfake_challenge
┣ 📂 iris/ # Core scripts
┃ ┣ 📜 predict_folder.py # Predict on a folder of videos
┃ ┣ 📜 kernel_utils.py # Utilities: video reader, face extractor, etc.
┃ ┣ 📂 configs/ # Model configs (b5.json, b7.json, etc.)
┃ ┗ 📂 libs/ # Extra libs (landmark detectors, etc.)
┣ 📂 weights/ # Model weights (downloaded via script)
┣ 📂 logs/ # Training/inference logs
┣ 📂 images/ # Visualizations (augmentations, loss plots, etc.)
┣ 📜 download_weights.sh # Script to download pretrained weights
┣ 📜 predict_submission.sh # Run inference on test set
┗ 📜 README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
2️⃣ Install Dependencies
It’s recommended to use a virtual environment.

bash
Copy
Edit
pip install -r requirements.txt
If you see ModuleNotFoundError, install missing packages manually:

bash
Copy
Edit
pip install albumentations opencv-python dlib torch torchvision
3️⃣ Download Pretrained Weights
bash
Copy
Edit
bash download_weights.sh
This will create a weights/ folder with pretrained models.

▶️ Running Inference
You can run inference on a folder of videos using the preconfigured script:

bash
Copy
Edit
./predict_submission.sh /path/to/input_videos output.csv
/path/to/input_videos → directory containing .mp4 videos

output.csv → generated file with predictions (video, label)

Example:

bash
Copy
Edit
./predict_submission.sh ./test_videos submission.csv
🐳 Running with Docker (Optional)
Build the Docker image:

bash
Copy
Edit
docker build -t deepfake-detector .
Run inference inside the container:

bash
Copy
Edit
docker run -v $(pwd)/test_videos:/app/test_videos deepfake-detector ./predict_submission.sh /app/test_videos submission.csv
📝 Notes
The ensemble inference combines multiple models for better performance.

By default, the system uses face detection + augmentations before prediction.

Large video sets may take time due to face extraction & batching.

📊 Example Output
Example CSV (submission.csv):

csv
Copy
Edit
filename,label
video1.mp4,FAKE
video2.mp4,REAL
video3.mp4,FAKE
👨‍💻 Authors
Your Name (@yourgithub)

Contributors welcome! Please open issues and PRs.

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

yaml
Copy
Edit

---
