# 👁️ AI Eye Disease Detection

An end-to-end deep learning project that classifies four major eye diseases from retinal scans.
Built with TensorFlow/Keras and deployed through a modern, user-friendly Flask web interface.

---

## 🔍 Key Features

- 🌡️ **Disease Prediction with Confidence Score**
- 📊 **Grad-CAM Heatmap Visualization**
- 📄 **Auto-Generated PDF Report**
- 🖼️ **Real-time Image Preview**

---

## 📘 Project Overview

This project leverages **transfer learning** with the **Xception** architecture to classify retinal images into:

- **Cataract**
- **Diabetic Retinopathy**
- **Glaucoma**
- **Normal**

The trained model is deployed via a Flask web app that offers prediction, confidence score, explainability (XAI), and a downloadable diagnostic report.

---

## ✨ Features

- ✅ **Multi-Class Classification:** Four-class eye disease classifier
- ✅ **Modern UI:** Responsive and accessible diagnostics dashboard
- ✅ **Confidence Score:** Clean circular progress bar with % confidence
- ✅ **Model Explainability:** Grad-CAM to show model’s attention area
- ✅ **Downloadable Reports:** PDF report generation per diagnosis
- ✅ **Image Preview:** See the selected image before upload

---

## 🧠 Technologies Used

- Python 3.12
- Flask
- TensorFlow / Keras (Xception)
- OpenCV & Pillow (Image processing)
- Grad-CAM (Explainable AI)
- ReportLab (PDF generation)
- HTML, CSS, JavaScript (Frontend)

---

## 🧪 Model Summary

- Trained on: Labeled retinal disease dataset
- Input size: **299x299**
- Model: **Xception**
- Framework: **Keras** with early stopping and image augmentation

---

## 🚀 User Flow

1. Upload a retina image (JPEG, PNG).
2. Model predicts one of the diseases.
3. Grad-CAM heatmap is generated.
4. Prediction + confidence is displayed.
5. PDF report is auto-generated and available for download.

---

## 📂 Project Structure





Eye_Disease_Detection/
├── app.py                  # Flask backend
├── models/
│   └── xception_model.h5   # Trained Keras model
├── static/
│   ├── uploads/            # Uploaded images
│   ├── heatmaps/           # Grad-CAM visualizations
│   ├── reports/            # Generated PDF reports
│   └── styles.css          # Custom CSS
├── templates/
│   ├── index.html          # Home page UI
│   └── result.html         # Prediction result view
├── requirements.txt       
└── README.md               






























---

🛠️ **How to Run Locally:**

git clone [https://github.com/vvssmohan/Eye_Disease_Detection_Using-Deep_Learning.git]


# Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Run the app:
python app.py
Open your browser and visit: http://127.0.0.1:5000/

🙌 Author

Name: Veera Venkata Sesha Sai Mohan

GitHub: @vvssmohan

Mail: ramanammohan12@gmail.com







