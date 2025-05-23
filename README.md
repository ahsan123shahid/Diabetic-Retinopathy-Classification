# 🧠 Retinopathy Classifier Web App

This is a Flask-based web application that classifies retinal images to detect **diabetic retinopathy** using a trained Artificial Neural Network (ANN) model.

## 🚀 Features

- Upload retina images (JPG, PNG)
- Predict retinopathy severity using a deep learning model
- Display prediction results in your browser

## 🧰 Tech Stack

- Python
- Flask
- TensorFlow / Keras
- HTML / CSS (Jinja2 templates)

## 📦 Project Structure
retinopathy-classifier/
├── static/ # CSS, JS, and images
├── templates/ # HTML templates (Flask)
├── uploads/ # Uploaded retina images
├── app.py # Flask web server
├── requirements.txt # Python dependencies
├── MY_MODEL.h5 # Trained ANN model (not on GitHub due to size)
The model file is not included due to size limitations. Please contact me or use your own trained model saved as MY_MODEL.h5.

pip install -r requirements.txt
python app.py
http://127.0.0.1:5000/

