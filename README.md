👁️ AI Eye Disease Detection
An end-to-end deep learning project that classifies four major eye diseases from retinal scans. The application is built with a TensorFlow/Keras backend and a modern,       user-friendly web interface using Flask
🔍 Disease Prediction with Confidence Score
🌡️ Grad-CAM Heatmap Visualisation
📄 Auto-Generated PDF Report

## Project Overview
This project leverages transfer learning with the Xception architecture to accurately classify retinal images into four categories: Cataract, Diabetic Retinopathy, Glaucoma, and Normal. The trained model is served through a Flask web application that provides a complete diagnostic report, including model explainability through Grad-CAM heatmaps.

Features ✨
Multi-Class Classification: Accurately classifies four different eye conditions.
Modern UI: A sleek, responsive, and user-friendly "diagnostics dashboard" interface.
Confidence Score: Displays the model's prediction confidence with a clean, circular progress bar.
Model Explainability (XAI): Generates Grad-CAM heatmaps to visualise which parts of the image the AI model focused on for its prediction.
PDF Reports: Automatically generates and allows downloading of a PDF report for each diagnosis.
Image Preview: Users can preview their selected image before uploading for analysis.

🧠 Technologies Used
Python 3.12
Flask
TensorFlow / Keras (Xception model)
OpenCV & Pillow
Grad-CAM (Explainable AI)
ReportLab (for PDF generation)
HTML, CSS, JavaScript (Frontend UI)

🧪 Model
Pretrained on a dataset of labelled retinal images
Image input size: 299x299
Model used: Xception
Trained using Keras with early stopping and augmentation

🚀 Features
✅ Upload eye images through an interactive web interface
✅ Model predicts one of the following diseases:
    Cataract
    Diabetic Retinopathy
    Glaucoma
    Normal
✅ Visual heatmap overlay (Grad-CAM)
✅ Confidence percentage
✅ Downloadable PDF diagnosis report

📂 Project Structure
Eye_Disease_Detection/
│
├── app.py                       # Flask backend
├── models/
│   └── xception_model.h5        # Trained Keras model
├── static/
│   ├── uploads/                 # Uploaded images
│   ├── heatmaps/                # Grad-CAM visualizations
│   ├── reports/                 # Generated PDF reports
│   └── styles.css               # Custom CSS
├── templates/
│   ├── index.html               # Home page UI
│   └── result.html              # Prediction result view
├── requirements.txt             # Python dependencies
└── README.md                    # You're reading this 😉


🛠️ How to Run Locally:

git clone https://github.com/vvssmohan/Eye_Disease_Detection_Using-Deep_Learning.git
cd Eye_Disease_Detection_Using-Deep_Learning

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies: pip install -r requirements.txt

Run the app: python app.py

Open your browser and visit:  http://127.0.0.1:5000/

🙌 Author
Name: Veera Venkata Sesha Sai Mohan
GitHub: @vvssmohan
Mail:ramanammohan12@gmail.com







