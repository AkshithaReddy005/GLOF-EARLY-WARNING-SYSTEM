# GLOF: Glacial Lake Outburst Flood Early Warning System

This project predicts Glacial Lake Outburst Floods (GLOFs) using real-time IoT sensor data and historical trends. It provides accurate early warning notifications through time-series data analysis and deep learning models.

## Features
- Predicts GLOF events using LSTM-based deep learning models
- Processes real-time and historical data
- Web interface for user input and result visualization (Flask)
- Model training, preprocessing, and prediction scripts

## Project Structure
- `main.py`: Flask web app entry point; routes for input, results, and static pages
- `modelpredict.py`: Model loading, preprocessing, and prediction logic
- `bilstmmodel.py` & `smot_model.py`: Data preprocessing and model training scripts
- `app.py`: (Legacy/alternate entry point)
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, images)
- `dataset/GLOFData.csv`: Main dataset used for model training
- `*.h5`, `*.pkl`: Trained model and preprocessing artifacts

## Setup Instructions
1. **Clone the repository:**
   ```sh
   git clone https://github.com/<YOUR-USERNAME>/<REPO-NAME>.git
   cd <REPO-NAME>
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```sh
   python main.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Usage
- Open the web interface and enter sensor data as a comma-separated string.
- The system will process the input and display GLOF prediction results.

## Requirements
- Python 3.7+
- Flask
- TensorFlow
- scikit-learn
- pandas, numpy, joblib

*See `requirements.txt` for full package list.*

## Credits
Developed by Akshitha and contributors. For questions, open an issue or contact via the repository.

---
*This project is for research and educational purposes.*
