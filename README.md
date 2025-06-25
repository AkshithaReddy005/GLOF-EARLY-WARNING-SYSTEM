<div align="center">

# 🌊 GLOF Early Warning System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask)](https://flask.palletsprojects.com/)

Predicting Glacial Lake Outburst Floods (GLOFs) using real-time IoT sensor data and historical trends, powered by deep learning and a user-friendly web interface.

</div>

---

## 🚀 Features

- 🤖 **LSTM-based Deep Learning Model** for accurate GLOF prediction
- 📊 Processes both real-time and historical data
- 🌐 **Web Interface** for easy data input & result visualization
- 🛠️ Model training, preprocessing, and prediction scripts

---

## 🗂️ Project Structure

| File/Folder           | Purpose                                                      |
|----------------------|--------------------------------------------------------------|
| `main.py`            | Flask web app entry point; routes for input, results, pages   |
| `modelpredict.py`    | Model loading, preprocessing, and prediction logic            |
| `bilstmmodel.py`     | Data preprocessing & model training script                   |
| `smot_model.py`      | Alternative data/model script                                |
| `app.py`             | (Legacy/alternate entry point)                               |
| `templates/`         | HTML templates for the web interface                         |
| `static/`            | Static files (CSS, images)                                   |
| `dataset/GLOFData.csv`| Main dataset used for model training                        |
| `*.h5`, `*.pkl`      | Trained model & preprocessing artifacts                      |

---

## ⚡ Quickstart

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AkshithaReddy005/GLOF-EARLY-WARNING-SYSTEM.git
   cd GLOF-EARLY-WARNING-SYSTEM
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```sh
   python main.py
   ```
   The app will be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 💻 Usage

- Open the web interface and enter sensor data as a comma-separated string.
- The system will process the input and display GLOF prediction results.

---

## 🧩 Requirements

- Python 3.7+
- Flask
- TensorFlow
- scikit-learn
- pandas, numpy, joblib

*See `requirements.txt` for the full package list.*

---

## 🙌 Credits

Developed by **Akshitha** and contributors. For questions, open an issue or contact via the repository.

---

<div align="center">
  <sub>🌏 This project is for research and educational purposes. Stay safe! 🌏</sub>
</div>
