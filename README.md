# 🌿 Plant Disease Detection System

A comprehensive machine learning-based web application for detecting plant diseases using three different prediction methods: text-based symptom analysis, quantitative feature analysis, and image-based diagnosis.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## 🎯 Overview

This application provides an AI-powered platform to detect plant diseases across **Tomato, Potato, and Pepper** plants using multiple prediction approaches:

- **Text-Based Prediction**: Analyze written symptom descriptions (48 models)
- **Quantitative Prediction**: Process numerical plant measurements (37 models)
- **Image-Based Prediction**: Analyze plant leaf images (5 deep learning models)

**Total Models**: 90 trained models  
**Disease Classes**: 15 different plant diseases

---

## ✨ Features

- **Multi-Model Support**: Choose from 90 different trained models
- **Three Prediction Methods**: Text, Quantitative, and Image analysis
- **User-Friendly Interface**: Clean, modern web interface with easy navigation
- **Real-Time Predictions**: Instant disease detection with confidence scores
- **Input Validation**: Automatic gibberish detection for text inputs
- **Comprehensive Results**: Disease name, confidence level, and model information

---

## 💻 System Requirements

### Minimum Requirements:
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher (3.9-3.11 recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space (for models and dependencies)
- **Internet**: Required for initial setup (downloading dependencies)

### Software Prerequisites:
- Python 3.8+
- pip (Python package manager)
- Git (optional, for cloning repository)

---

## 🚀 Installation Guide

Follow these steps carefully to set up the project on your machine:

### Step 1: Download the Project

**Option A - Using Git:**
```bash
git clone https://github.com/gowthamalasakaniS574661/GDP-TEAM2.git
cd GDP-TEAM2
```

**Option B - Download ZIP:**
1. Download the project ZIP file from the repository
2. Extract it to your desired location
3. Open terminal/command prompt in the extracted folder

### Step 2: Verify Python Installation

Check if Python is installed and its version:

```bash
# On macOS/Linux:
python3 --version

# On Windows:
python --version
```

**Expected Output**: `Python 3.x.x` (where x is 8 or higher)

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Install via Homebrew: `brew install python3`
- **Linux**: `sudo apt-get install python3 python3-pip`

### Step 3: Create a Virtual Environment (Recommended)

Creating a virtual environment keeps your project dependencies isolated:

```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

**You'll see `(venv)` prefix in your terminal when activated.**

### Step 4: Install Required Dependencies

Install all necessary Python packages:

```bash
# On macOS/Linux:
pip3 install -r requirements.txt

# On Windows:
pip install -r requirements.txt
```

**This will install:**
- Flask (web framework)
- TensorFlow (deep learning)
- scikit-learn (machine learning)
- Pandas, NumPy (data processing)
- Pillow, OpenCV (image processing)
- And other dependencies...

**Installation may take 5-15 minutes depending on your internet speed.**

### Step 5: Verify Model Files

Ensure all model directories exist with trained models:

```bash
# Check if model directories exist:
ls -la MLALGO/text_model_artifacts/
ls -la MLALGO/quant_model_artifacts/
ls -la MLALGO/image_models/
ls -la MLALGO/image_models_all/
```

**Expected Output**: You should see `.joblib` files and `.keras` files in respective directories.

If models are missing, ensure you have the complete project files including all subdirectories.

---

## 🎮 Running the Application

### Step 1: Navigate to Project Directory

```bash
cd /path/to/GDP-TEAM2
```

### Step 2: Activate Virtual Environment (if created)

```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Start the Flask Server

```bash
# On macOS/Linux:
python3 app.py

# On Windows:
python app.py
```

**Expected Output:**
```
 * Running on http://127.0.0.1:5001
 * Found 48 text models
 * Found 37 quant models  
 * Found 5 image models
 * WARNING: This is a development server. Do not use in production.
```

### Step 4: Access the Application

Open your web browser and visit:
```
http://localhost:5001
```

or

```
http://127.0.0.1:5001
```

**You should see the Plant Disease Detection System home page! 🎉**

---

## 📱 Using the Application

### Home Page
- You'll see three cards representing different prediction methods
- **Statistics**: 90 Models | 3 Types | 15 Disease Classes

### 1️⃣ Text-Based Prediction

1. Click on **"Text-Based Prediction"** card (📝)
2. Select a model from the dropdown (48 options)
3. Enter plant symptoms in the text area
   - Example: "leaves turning yellow with brown spots"
   - Example: "wilting plant with dark lesions"
4. Click **"Predict Disease"**
5. View results: Disease name, confidence percentage, model used

**Tips:**
- Use clear, descriptive symptom descriptions
- Include multiple symptoms for better accuracy
- Avoid random/gibberish text (system will reject it)

### 2️⃣ Quantitative Prediction

1. Click on **"Quantitative Prediction"** card (📊)
2. Select a model from the dropdown (37 options)
3. Enter numerical values for features
   - The form will show the required number of features based on the selected model
   - Example features: temperature, humidity, pH levels, etc.
4. Click **"Predict Disease"**
5. View results with confidence score

**Tips:**
- Ensure all feature values are numeric
- Different models require different numbers of features
- The feature count updates automatically when you select a model

### 3️⃣ Image-Based Prediction

1. Click on **"Image-Based Prediction"** card (🖼️)
2. Select a model from the dropdown (5 options)
3. Upload a plant leaf image
   - Click the upload area or drag & drop
   - Supported formats: JPG, PNG
   - Maximum size: 10MB
4. Preview your uploaded image
5. Click **"Predict Disease"**
6. View results: Disease name, confidence, prediction index

**Tips:**
- Use clear, well-lit images of plant leaves
- Avoid blurry or very dark images
- Center the diseased area in the image
- Higher resolution images work better

### 🔙 Navigation
- Every prediction page has a **"Back to Home"** button
- Return to home page to try different prediction methods

---

## 📂 Project Structure

```
GDP-TEAM2/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── templates/                      # HTML templates
│   ├── home.html                  # Landing page
│   ├── text.html                  # Text prediction page
│   ├── quant.html                 # Quantitative prediction page
│   └── image.html                 # Image prediction page
│
├── MLALGO/                        # Model artifacts directory
│   ├── text_model_artifacts/      # 48 text-based models (.joblib)
│   │   ├── tfidf.joblib
│   │   ├── label_encoder.joblib
│   │   └── [model files...]
│   │
│   ├── quant_model_artifacts/     # 37 quantitative models (.joblib)
│   │   └── [model files...]
│   │
│   ├── image_models/              # Original image model
│   │   └── model/
│   │       ├── saved_model.pb
│   │       └── class_indices.json
│   │
│   └── image_models_all/          # Additional 4 Keras models
│       ├── Xception.keras
│       ├── extractor.keras
│       ├── m.keras
│       └── model.keras
│
├── plant_disease_data.csv         # Training dataset
└── [other files...]
```

---

## 🔧 Troubleshooting

### Problem 1: Port Already in Use
**Error**: `Address already in use. Port 5001 is in use`

**Solution**:
```bash
# On macOS/Linux:
lsof -ti:5001 | xargs kill -9

# On Windows:
netstat -ano | findstr :5001
taskkill /PID <PID_NUMBER> /F
```

### Problem 2: Module Not Found
**Error**: `ModuleNotFoundError: No module named 'flask'` (or other module)

**Solution**:
```bash
# Reinstall dependencies:
pip3 install -r requirements.txt --force-reinstall
```

### Problem 3: TensorFlow Import Error
**Error**: `ImportError: cannot import name 'tensorflow'`

**Solution**:
```bash
# Install TensorFlow separately:
pip3 install tensorflow>=2.13.0
```

### Problem 4: Permission Denied
**Error**: `Permission denied` when running app.py

**Solution**:
```bash
# Make the file executable:
chmod +x app.py

# Or run with Python explicitly:
python3 app.py
```

### Problem 5: Models Not Loading
**Error**: "No models found" or "Model not found"

**Solution**:
- Verify all model directories exist
- Check file permissions: `ls -la MLALGO/`
- Ensure model files (.joblib, .keras) are present
- Re-download project files if missing

### Problem 6: Image Upload Fails
**Error**: Image prediction returns error

**Solution**:
- Check image format (use JPG or PNG only)
- Reduce image size (under 10MB)
- Ensure TensorFlow is properly installed
- Try a different model from the dropdown

### Problem 7: Low Confidence Warnings
**Message**: "Model confidence is low for this input"

**Solution**:
- Provide more detailed symptom descriptions
- Try a different model
- Ensure input data is relevant to plant diseases
- For images: use clearer, better-lit photos

### Problem 8: Gibberish Detection
**Error**: "Please enter valid text"

**Solution**:
- This is intentional validation
- Enter meaningful plant symptom descriptions
- Use proper words (not random characters)
- Example: "yellow spots on leaves" instead of "jhdsvcjhs"

---

## 🔬 Technical Details

### Disease Classes (15 Total)

**Pepper:**
1. Pepper__bell___Bacterial_spot
2. Pepper__bell___healthy

**Potato:**
3. Potato___Early_blight
4. Potato___Late_blight
5. Potato___healthy

**Tomato:**
6. Tomato_Bacterial_spot
7. Tomato_Early_blight
8. Tomato_Late_blight
9. Tomato_Leaf_Mold
10. Tomato_Septoria_leaf_spot
11. Tomato_Spider_mites_Two_spotted_spider_mite
12. Tomato__Target_Spot
13. Tomato__Tomato_YellowLeaf__Curl_Virus
14. Tomato__Tomato_mosaic_virus
15. Tomato_healthy

### Model Types

#### Text Models (48 total)
- **Algorithms**: BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, Logistic Regression, KNN, SGD
- **Feature Selection**: K-Best (MI), Percentile, SelectFromModel, Variance Threshold, SVD
- **Vectorization**: TF-IDF

#### Quantitative Models (37 total)
- **Algorithms**: Various scikit-learn classifiers
- **Input**: Numeric feature vectors
- **Feature Count**: Varies by model (displayed dynamically)

#### Image Models (5 total)
- **Framework**: TensorFlow/Keras
- **Models**: Custom CNN, Xception, Feature Extractor variants
- **Input Size**: 224x224 or 256x256 RGB images
- **Output**: 15-class predictions

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/text` | GET | Text prediction page |
| `/quant` | GET | Quantitative prediction page |
| `/image` | GET | Image prediction page |
| `/api/models` | GET | List text models |
| `/api/predict` | POST | Text prediction |
| `/api/quant_models` | GET | List quant models |
| `/api/predict_quant` | POST | Quant prediction |
| `/api/image_models` | GET | List image models |
| `/api/predict_image` | POST | Image prediction |

### Input Validation

**Text Inputs:**
- Minimum 5 characters
- At least 15% vowel ratio
- Maximum 5 consecutive consonants
- No excessive character repetition
- Rejects gibberish automatically

**Quantitative Inputs:**
- All numeric values
- Matches model's expected feature count
- Range validation per feature

**Image Inputs:**
- Formats: JPG, PNG
- Maximum size: 10MB
- Automatically resized to model input dimensions

---

## 🎓 Development Information

### Technologies Used
- **Backend**: Python 3.x, Flask 3.x
- **ML Framework**: scikit-learn 1.7.2
- **Deep Learning**: TensorFlow 2.13+
- **Image Processing**: Pillow, OpenCV
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

### Model Training
- Models were trained on plant disease datasets
- Feature engineering applied for text and quantitative models
- Transfer learning used for image models
- Cross-validation performed for model selection

### Future Enhancements
- [ ] Mobile application
- [ ] Real-time camera integration
- [ ] Treatment recommendations
- [ ] Multi-language support
- [ ] Batch image processing
- [ ] Model performance comparison
- [ ] User authentication
- [ ] Prediction history

---

## 📞 Support

If you encounter issues not covered in this guide:

1. Check the terminal output for error messages
2. Review the troubleshooting section above
3. Ensure all dependencies are correctly installed
4. Verify Python version compatibility
5. Check that all model files are present

---

## 📄 License

This project is part of GDP-TEAM2 coursework.

---

## 👥 Contributors

- GDP-TEAM2
- Repository: gowthamalasakaniS574661/GDP-TEAM2

---

## 🙏 Acknowledgments

- Plant Village Dataset
- TensorFlow Team
- scikit-learn Community
- Flask Framework

---

**Last Updated**: December 4, 2025

**Version**: 1.0.0

---

## Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/gowthamalasakaniS574661/GDP-TEAM2.git
cd GDP-TEAM2
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip3 install -r requirements.txt

# Run
python3 app.py

# Access
# Open browser: http://localhost:5001
```

---

**Happy Disease Detection! 🌱🔬**
