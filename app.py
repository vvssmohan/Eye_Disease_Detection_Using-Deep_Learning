import os
import logging
import numpy as np
import datetime
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
from reportlab.pdfgen import canvas

# Setup
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
REPORT_FOLDER = "static/reports"
MODEL_PATH = "models/eye_disease_model.h5"
CATEGORIES = ["Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"]
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

logging.basicConfig(level=logging.INFO)
model = load_model(MODEL_PATH)

def allowed_file(filename):
    
    """ Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_last_conv_layer_name(model):
    
    """ Finds the last convolutional layer in a Keras model for Grad-CAM."""
    for layer in reversed(model.layers):
        # Use layer.output.shape for robust compatibility across TensorFlow versions
        if isinstance(layer, tf.keras.layers.Conv2D) and len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("Could not find a Conv2D layer in the model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """ Generates a Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def generate_pdf_report(report_base_filename, prediction, confidence, image_path):
    
    """ Generates a PDF report of the diagnosis."""
    pdf_path = os.path.join(REPORT_FOLDER, f"{report_base_filename}.pdf")
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Eye Disease Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 770, f"Prediction: {prediction}")
    c.drawString(100, 750, f"Confidence: {confidence:.2f}%")
    c.drawString(100, 730, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawImage(image_path, 100, 500, width=200, height=200)
    c.save()
    
    return pdf_path

@app.route("/", methods=["GET", "POST"])
def index():
    
    """ Handles the main page, file uploads, and predictions."""
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == '' or not allowed_file(file.filename):
            flash("Invalid file or no file selected. Please upload a valid image (png, jpg, jpeg).")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess the image for the model
        img = image.load_img(filepath, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions)
        confidence = float(predictions[0][pred_index]) * 100
        label = CATEGORIES[pred_index]

        # Generate Grad-CAM heatmap
        last_conv_layer_name = find_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

        # Superimpose heatmap on the original image
        colored_heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)
        img_original = Image.open(filepath).resize((299, 299))
        heatmap_resized = Image.fromarray(colored_heatmap).resize((299, 299))
        superimposed_img = Image.blend(img_original.convert("RGB"), heatmap_resized, alpha=0.4)

        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        superimposed_img.save(heatmap_path)

        # Generate PDF report
        report_base_name = f"report_{os.path.splitext(filename)[0]}"
        report_path = generate_pdf_report(report_base_name, label, confidence, filepath)
        report_filename = os.path.basename(report_path)

        # Create relative paths for use in the HTML template
        template_img_path = f"uploads/{filename}"
        template_heatmap_path = f"heatmaps/{heatmap_filename}"

        return render_template("result.html",
                               result=label,
                               confidence=confidence,
                               img_path=template_img_path,
                               heatmap_path=template_heatmap_path,
                               report_filename=report_filename)

    return render_template("index.html")

@app.route("/download/<path:filename>")
def download(filename):
    
    """ Handles the downloading of PDF reports."""
    return send_from_directory(
        directory=REPORT_FOLDER,
        path=filename,
        as_attachment=True
    )

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(debug=True)
