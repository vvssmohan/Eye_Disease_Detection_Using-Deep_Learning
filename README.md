👁️ AI Eye Disease Detection
An end-to-end deep learning project that classifies four major eye diseases from retinal scans. The application is built with a TensorFlow/Keras backend and a modern, user-friendly web interface using Flask.

(This is a sample screenshot of the final UI. You can replace it with your own.)

## Project Overview
This project leverages transfer learning with the Xception architecture to accurately classify retinal images into four categories: Cataract, Diabetic Retinopathy, Glaucoma, and Normal. The trained model is served through a Flask web application that provides a complete diagnostic report, including model explainability through Grad-CAM heatmaps.

## Features ✨
Multi-Class Classification: Accurately classifies four different eye conditions.

Modern UI: A sleek, responsive, and user-friendly "diagnostics dashboard" interface.

Confidence Score: Displays the model's prediction confidence with a clean, circular progress bar.

Model Explainability (XAI): Generates Grad-CAM heatmaps to visualize which parts of the image the AI model focused on for its prediction.

PDF Reports: Automatically generates and allows downloading of a PDF report for each diagnosis.

Image Preview: Users can preview their selected image before uploading for analysis.

## Technology Stack 🛠️
Backend: Python, Flask

Deep Learning: TensorFlow, Keras

Image Processing: Pillow, OpenCV (implicitly used by TensorFlow)

Data Handling: NumPy

Reporting: ReportLab

Plotting/Evaluation: Matplotlib, Scikit-learn

## Project Structure 📂
📁 Eye_Disease_Project/
├── 📄 app.py                  # Main Flask application
├── 📄 model_training.py       # Script to train the deep learning model
├── 📄 requirements.txt        # Project dependencies
├── 📄 README.md               # This file
├── 📁 dataset/                # Folder for the image dataset
│   ├── 📁 Cataract/
│   └── ... (other categories)
├── 📁 models/                 # Saved Keras model appears here after training
│   └── 📄 eye_disease_model.h5
├── 📁 static/
│   ├── 📁 images/
│   │   └── 📄 background.jpg
│   ├── 📄 styles.css
│   ├── 📁 uploads/
│   ├── 📁 heatmaps/
│   └── 📁 reports/
└── 📁 templates/
    ├── 📄 index.html
    └── 📄 result.html
## Setup and Installation 🚀
Follow these steps to set up and run the project locally.

### 1. Prerequisites
Python 3.8+

Anaconda or another virtual environment manager (recommended).

### 2. Clone the Repository
Bash

git clone <your-repository-link>
cd Eye_Disease_Project
### 3. Set Up the Dataset
Download the dataset from Kaggle: Eye Diseases Classification Dataset

Unzip the file.

Place the contents (folders named Cataract, Glaucoma, etc.) inside the dataset/ directory in your project folder.

### 4. Create a Virtual Environment and Install Dependencies
It's highly recommended to use a virtual environment to keep dependencies isolated.

Bash

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages from requirements.txt
pip install -r requirements.txt
You will need to create a requirements.txt file with the following content:

Plaintext

tensorflow
flask
numpy
Pillow
matplotlib
reportlab
scikit-learn
## How to Run the Project 🏃‍♂️
The project runs in two stages: training the model and then running the web application.

### Stage 1: Train the Model
First, you need to run the training script to generate the eye_disease_model.h5 file.

Bash

python model_training.py
This process may take some time depending on your hardware. It will create the model file inside the models/ folder.

### Stage 2: Run the Flask Web Application
Once the model is trained and saved, you can start the web server.

Bash

flask run
Or alternatively:

Bash

python app.py
Now, open your web browser and navigate to http://127.0.0.1:5000 to use the application.