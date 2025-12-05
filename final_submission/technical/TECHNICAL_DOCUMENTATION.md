# Technical Implementation Documentation

## Plant Disease Detection System - Technical Details

### Author: GDP-TEAM2
### Date: December 5, 2025

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Implementation Details](#implementation-details)
3. [API Specification](#api-specification)
4. [Model Information](#model-information)
5. [Code Structure](#code-structure)
6. [Setup & Configuration](#setup--configuration)
7. [Deployment Guide](#deployment-guide)

---

## Architecture Overview

### System Architecture

The application follows a **Model-View-Controller (MVC)** pattern with a Flask-based REST API backend:

```
┌─────────────────────────────────────────────────────┐
│                  Client Layer                       │
│     (HTML5 + CSS3 + Vanilla JavaScript)             │
│  ┌──────────────────────────────────────────────┐  │
│  │  home.html | text.html | quant.html | image  │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/HTTPS
                     │ REST API Calls
                     ▼
┌─────────────────────────────────────────────────────┐
│              Application Server (Flask)             │
│  ┌──────────────────────────────────────────────┐  │
│  │   app.py (Main Application - 1231 lines)     │  │
│  │   - Route handlers (@app.route)              │  │
│  │   - API endpoints (/api/*)                   │  │
│  │   - Model loaders & adapters                 │  │
│  │   - Prediction logic                         │  │
│  │   - Input validation                         │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               Model Layer (ML)                      │
│  ┌──────────┬──────────────┬────────────────────┐  │
│  │  Text    │   Quant      │    Image           │  │
│  │  Models  │   Models     │    Models          │  │
│  │  (48)    │   (37)       │    (5)             │  │
│  │          │              │                    │  │
│  │ sklearn  │  sklearn     │  TensorFlow/Keras  │  │
│  │ joblib   │  joblib      │  .keras / .pb      │  │
│  └──────────┴──────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Technology Stack

#### Backend Stack
- **Python 3.9+**: Core programming language
- **Flask 3.x**: Lightweight WSGI web framework
- **Werkzeug**: WSGI utility library (Flask dependency)
- **Jinja2**: Template engine for HTML rendering

#### Machine Learning Stack
- **scikit-learn 1.7.2**: Classical ML algorithms
  - Naive Bayes (Bernoulli, Complement, Multinomial)
  - Linear SVC
  - Logistic Regression
  - K-Nearest Neighbors
  - SGD Classifier
- **TensorFlow 2.13+**: Deep learning framework
- **Keras**: High-level neural network API
- **joblib**: Model serialization/deserialization

#### Data Processing Stack
- **NumPy 1.24+**: Numerical computations
- **Pandas 2.0+**: Data manipulation
- **Pillow 10.0+**: Image processing
- **OpenCV 4.8+**: Computer vision operations

#### Frontend Stack
- **HTML5**: Semantic markup
- **CSS3**: Styling with gradients, flexbox, grid
- **JavaScript ES6+**: Asynchronous API calls, DOM manipulation
- **Fetch API**: AJAX requests

---

## Implementation Details

### File Structure

```
final_submission/technical/
│
├── app.py (1231 lines)
│   ├── Imports & Configuration (lines 1-50)
│   ├── Helper Functions (lines 51-320)
│   ├── Model Discovery Functions (lines 321-850)
│   ├── API Endpoints (lines 851-1200)
│   └── Main Entry Point (lines 1201-1231)
│
├── requirements.txt
│   └── All Python dependencies with versions
│
└── templates/
    ├── home.html (190 lines)
    ├── text.html (275 lines)
    ├── quant.html (305 lines)
    ├── image.html (285 lines)
    └── index.html (736 lines - legacy)
```

### Core Application Code (app.py)

#### 1. Configuration & Constants

```python
# Lines 10-25: Directory configuration
ART_DIR = "MLALGO/text_model_artifacts"
QUANT_MODELS_DIR = "MLALGO/quant_model_artifacts"
IMAGE_MODELS_DIR = "MLALGO/image_models"
IMAGE_MODELS_DIR_ALL = "MLALGO/image_models_all"

# Lines 30-35: Application settings
CONFIDENCE_THRESHOLD = 0.3
MODEL_CACHE_TTL = 10.0  # seconds
DEBUG_MODE = False
```

#### 2. Helper Functions

**Gibberish Detection** (lines 224-290)
```python
def is_gibberish(text: str) -> bool:
    """
    Detects invalid/random text input using heuristics:
    - Minimum length check (5 chars)
    - Vowel ratio analysis (≥15%)
    - Consonant run detection (max 5 consecutive)
    - Alphabetic character ratio check (≥50%)
    """
```

**Confidence Computation** (lines 292-320)
```python
def compute_confidence(model, X):
    """
    Extracts confidence score from model prediction:
    - predict_proba() for probabilistic models
    - decision_function() + softmax for SVM/Linear models
    - Returns float in [0, 1] range
    """
```

#### 3. Model Loading

**Text Model Loader** (lines 400-450)
```python
def load_vectorizer_for_model(model_filename):
    """
    Loads appropriate TF-IDF vectorizer for model:
    - Checks model-specific vectorizer first
    - Falls back to generic tfidf.joblib
    - Returns (vectorizer_object, vectorizer_name)
    """
```

**Image Model Adapter** (lines 520-600)
```python
class SimpleKerasAdapter:
    """
    Wrapper for standalone .keras models:
    - Lazy loading (load on first prediction)
    - Automatic input preprocessing
    - Label mapping to disease names
    """
```

#### 4. API Endpoints

**Text Prediction** (lines 1077-1205)
```python
@app.route("/api/predict", methods=["POST"])
def api_predict():
    # 1. Parse JSON request
    # 2. Load model and vectorizer
    # 3. Validate input (gibberish check)
    # 4. Transform text with TF-IDF
    # 5. Check feature dimensions
    # 6. Make prediction
    # 7. Compute confidence
    # 8. Return JSON response
```

**Quantitative Prediction** (lines 809-870)
```python
@app.route("/api/predict_quant", methods=["POST"])
def api_predict_quant():
    # 1. Parse request (supports 'model' or 'model_name')
    # 2. Load model and label encoder
    # 3. Validate feature count
    # 4. Convert features to numpy array
    # 5. Make prediction
    # 6. Map to disease name
    # 7. Compute confidence
```

**Image Prediction** (lines 353-520)
```python
@app.route("/api/predict_image", methods=["POST"])
def api_predict_image():
    # 1. Receive multipart form data
    # 2. Validate image format
    # 3. Load selected Keras model
    # 4. Preprocess image (resize, normalize)
    # 5. Run inference
    # 6. Map prediction to disease label
    # 7. Return results
```

### Frontend Implementation

#### JavaScript Architecture

Each HTML page uses a consistent pattern:

```javascript
// 1. Page Load: Fetch available models
fetch('/api/models')
    .then(r => r.json())
    .then(data => populateDropdown(data.models));

// 2. User Input: Collect form data
const formData = {
    model_filename: selectedModel,
    text: userInput
};

// 3. API Call: Send prediction request
fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData)
})
.then(r => r.json())
.then(displayResults);

// 4. Display: Show results with confidence bar
document.getElementById('prediction').textContent = data.prediction;
updateConfidenceBar(data.confidence);
```

#### CSS Design System

**Color Variables** (consistent across all pages)
```css
:root {
    --bg1: #03031a;        /* Dark blue-black */
    --bg2: #071124;        /* Lighter dark blue */
    --accent: #6ef0c3;     /* Cyan-green */
    --accent2: #5a8bff;    /* Sky blue */
    --text: #e6f7ff;       /* Light text */
    --muted: #9fbfaa;      /* Muted gray-green */
    --success: #caffdf;    /* Success green */
    --danger: #ff6b6b;     /* Error red */
}
```

**Responsive Grid** (used in home.html)
```css
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 25px;
}
```

---

## API Specification

### Base URL
```
http://localhost:5001
```

### Endpoints

#### 1. GET `/`
**Purpose**: Landing page  
**Response**: HTML page  
**Status Code**: 200

#### 2. GET `/text`
**Purpose**: Text prediction interface  
**Response**: HTML page  
**Status Code**: 200

#### 3. GET `/quant`
**Purpose**: Quantitative prediction interface  
**Response**: HTML page  
**Status Code**: 200

#### 4. GET `/image`
**Purpose**: Image prediction interface  
**Response**: HTML page  
**Status Code**: 200

#### 5. GET `/api/models`
**Purpose**: List text-based models  
**Response**:
```json
{
  "models": [
    {
      "filename": "kbest_mi_500__LinearSVC.joblib",
      "size_bytes": 15234,
      "vectorizer": "tfidf.joblib"
    }
  ]
}
```
**Status Code**: 200

#### 6. POST `/api/predict`
**Purpose**: Text-based disease prediction  
**Request Headers**: `Content-Type: application/json`  
**Request Body**:
```json
{
  "model_filename": "kbest_mi_500__LinearSVC.joblib",
  "text": "leaves turning yellow with brown spots"
}
```
**Success Response (200)**:
```json
{
  "prediction": "Tomato_Early_blight",
  "confidence": 0.87,
  "used_model": "kbest_mi_500__LinearSVC.joblib",
  "vectorizer_used": "tfidf.joblib"
}
```
**Error Responses**:
- **400 - Gibberish**: `{"error": "gibberish", "message": "Please enter valid text..."}`
- **400 - No Text**: `{"error": "No text provided"}`
- **400 - No Model**: `{"error": "No model_filename specified"}`
- **404 - Model Not Found**: `{"error": "Model not found: filename"}`
- **500 - Server Error**: `{"error": "error message"}`

#### 7. GET `/api/quant_models`
**Purpose**: List quantitative models  
**Response**:
```json
{
  "models": [
    {
      "filename": "quant_model.joblib",
      "size_bytes": 12345,
      "expected_features": 4
    }
  ]
}
```
**Status Code**: 200

#### 8. POST `/api/predict_quant`
**Purpose**: Quantitative prediction  
**Request Headers**: `Content-Type: application/json`  
**Request Body** (supports both field names):
```json
{
  "model": "quant_model.joblib",
  "features": [25.5, 60.3, 7.2, 15.8]
}
```
OR
```json
{
  "model_name": "quant_model.joblib",
  "features": [25.5, 60.3, 7.2, 15.8]
}
```
**Success Response (200)**:
```json
{
  "prediction": "Potato___Late_blight",
  "confidence": 0.92,
  "model_used": "quant_model.joblib"
}
```
**Error Responses**:
- **400 - No Model**: `{"error": "No model specified"}`
- **400 - No Features**: `{"error": "No features array provided"}`
- **400 - Feature Mismatch**: `{"error": "Expected X features, got Y"}`
- **404 - Model Not Found**: `{"error": "Model file not found"}`

#### 9. GET `/api/image_models`
**Purpose**: List image models  
**Response**:
```json
{
  "models": [
    {
      "name": "Xception.keras",
      "path": "/path/to/model",
      "labels": ["disease1", "disease2", ...]
    }
  ]
}
```
**Status Code**: 200

#### 10. POST `/api/predict_image`
**Purpose**: Image-based prediction  
**Request**: Multipart form data
- **image**: Image file (JPG/PNG)
- **model_name**: Selected model name

**Success Response (200)**:
```json
{
  "prediction": "Tomato_Late_blight",
  "confidence": 0.95,
  "model": "Xception.keras",
  "prediction_index": 7
}
```
**Error Responses**:
- **400 - No Image**: `{"error": "No image file provided"}`
- **400 - No Model**: `{"error": "No model_name specified"}`
- **404 - Model Not Found**: `{"error": "Model not found"}`
- **500 - Prediction Error**: `{"error": "error message"}`

---

## Model Information

### Text Models (48 Total)

#### Feature Selection Methods
1. **K-Best Mutual Information (500 features)**
   - Models: BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, Logistic_reg, KNN_5, SGD_log
   - File pattern: `kbest_mi_500__*.joblib`

2. **Percentile Mutual Information (20%)**
   - Models: Same as above
   - File pattern: `percentile_mi_20__*.joblib`

3. **SelectFromModel with ExtraTrees**
   - Models: Same as above
   - File pattern: `selectfrom_extratrees__*.joblib`

4. **Variance Threshold**
   - Models: Same as above
   - File pattern: `variance__*.joblib`

5. **SVD Dense (100 components)**
   - Models: ExtraTrees_100, LightGBM_100, Logistic_lbfgs, RandomForest_100
   - File pattern: `svd_dense_100__*.joblib`

#### Vectorizers
- `tfidf.joblib` - Generic TF-IDF vectorizer
- `kbest_mi_500_tfidf.joblib` - For K-Best models
- `percentile_mi_20_tfidf.joblib` - For Percentile models
- `selectfrom_extratrees_tfidf.joblib` - For SelectFromModel models
- `variance_tfidf.joblib` - For Variance models
- `svd_dense_100_tfidf.joblib` - For SVD models

### Quantitative Models (37 Total)

#### Algorithms Used
- AdaBoost
- Decision Tree
- Extra Trees
- Gaussian Naive Bayes
- Gradient Boosting
- K-Nearest Neighbors
- Logistic Regression
- Random Forest
- SGD Classifier
- SVC (Support Vector Classifier)

#### Feature Engineering
- Standard Scaling
- One-Hot Encoding
- Feature Selection (SelectKBest, SelectFromModel, Percentile)

### Image Models (5 Total)

#### 1. Original Model (saved_model.pb)
- **Architecture**: Custom CNN
- **Input Size**: 256x256 RGB
- **Output Classes**: 15
- **Format**: TensorFlow SavedModel
- **Location**: `MLALGO/image_models/model/`

#### 2. Xception.keras
- **Architecture**: Xception (Transfer Learning)
- **Size**: ~88 MB
- **Input Size**: 299x299 RGB
- **Preprocessing**: Xception-specific normalization

#### 3. extractor.keras
- **Architecture**: Feature extractor variant
- **Size**: ~88 MB
- **Input Size**: 224x224 RGB

#### 4. m.keras
- **Architecture**: Custom model
- **Size**: ~299 MB
- **Input Size**: 256x256 RGB

#### 5. model.keras
- **Architecture**: General CNN
- **Size**: ~270 MB
- **Input Size**: 256x256 RGB

---

## Code Structure

### Key Functions & Classes

#### Input Validation
- `is_gibberish(text)` - Detects invalid text input
- `validate_features(features, expected)` - Checks feature count

#### Model Management
- `discover_text_models()` - Scans text model directory
- `discover_quant_models()` - Scans quant model directory
- `discover_image_models()` - Scans both image directories
- `get_label_encoder()` - Lazy loads label encoder

#### Prediction Pipeline
- `load_vectorizer_for_model(model_filename)` - Loads TF-IDF vectorizer
- `compute_confidence(model, X)` - Extracts confidence score
- `preprocess_image(image, target_size)` - Prepares image for model

#### Adapters & Wrappers
- `SimpleKerasAdapter` - Wraps .keras files
- `KerasModelAdapter` - Wraps SavedModel format

### Error Handling Strategy

1. **Try-Except Blocks**: All API endpoints wrapped
2. **Specific Error Messages**: Clear, actionable feedback
3. **HTTP Status Codes**: Proper RESTful codes
4. **Logging**: `traceback.print_exc()` for debugging
5. **Fallback Mechanisms**: Default values when possible

### Performance Optimizations

1. **Lazy Loading**: Models loaded only when needed
2. **Caching**: Image model list cached (10s TTL)
3. **Streaming**: Large files processed in chunks
4. **Minimal Dependencies**: Only load TensorFlow when needed

---

## Setup & Configuration

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/gowthamalasakaniS574661/GDP-TEAM2.git
cd GDP-TEAM2
```

2. **Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip3 install -r requirements.txt
```

4. **Verify Model Files**
```bash
ls MLALGO/text_model_artifacts/
ls MLALGO/quant_model_artifacts/
ls MLALGO/image_models/
ls MLALGO/image_models_all/
```

5. **Run Application**
```bash
python3 app.py
```

### Configuration Options

**Environment Variables** (optional):
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
export MODEL_CACHE_TTL=60
```

**Port Configuration** (app.py line 1230):
```python
app.run(host="127.0.0.1", port=5001, debug=False)
```

---

## Deployment Guide

### Development Deployment

```bash
# Standard development server
python3 app.py
```

### Production Deployment

#### Option 1: Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

#### Option 2: uWSGI

```bash
# Install uWSGI
pip install uwsgi

# Run with 4 processes
uwsgi --http 0.0.0.0:5001 --wsgi-file app.py --callable app --processes 4
```

#### Option 3: Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python3", "app.py"]
```

```bash
# Build and run
docker build -t plant-disease-detection .
docker run -p 5001:5001 plant-disease-detection
```

### Security Considerations

1. **HTTPS**: Use reverse proxy (nginx) for SSL
2. **Input Sanitization**: Already implemented
3. **Rate Limiting**: Add Flask-Limiter
4. **CORS**: Configure for production domains
5. **File Upload Limits**: Already set (10MB)
6. **Authentication**: Add JWT if needed

### Performance Tuning

1. **Model Caching**: Increase cache TTL for production
2. **Connection Pooling**: Use database for predictions history
3. **CDN**: Serve static assets from CDN
4. **Load Balancing**: Use nginx upstream for multiple instances
5. **Monitoring**: Add Prometheus metrics

---

## Testing

### Manual Testing Checklist

#### Text Prediction
- [ ] Valid symptom description → Correct prediction
- [ ] Gibberish input → Error message
- [ ] All 48 models → Load successfully
- [ ] Confidence scores → In [0, 1] range

#### Quantitative Prediction
- [ ] Correct features → Successful prediction
- [ ] Wrong feature count → Error message
- [ ] All 37 models → Load successfully
- [ ] Both `model` and `model_name` fields → Work

#### Image Prediction
- [ ] Valid JPG → Correct prediction
- [ ] Valid PNG → Correct prediction
- [ ] Large image → Size validation
- [ ] All 5 models → Return disease names

### Automated Testing (Future)

```python
# Example unit test
def test_gibberish_detection():
    assert is_gibberish("jhcvshjdsvcjhs") == True
    assert is_gibberish("leaves turning yellow") == False

def test_api_predict_text():
    response = client.post('/api/predict', json={
        'model_filename': 'kbest_mi_500__LinearSVC.joblib',
        'text': 'yellow leaves with spots'
    })
    assert response.status_code == 200
    assert 'prediction' in response.json
```

---

## Maintenance

### Regular Tasks
1. Update dependencies: `pip install -U -r requirements.txt`
2. Check model files integrity
3. Review error logs
4. Monitor disk space (models ~2GB)
5. Backup model files regularly

### Troubleshooting

**Common Issues**:
1. Port conflicts → Kill process on 5001
2. Model not found → Check file paths
3. Low memory → Reduce concurrent requests
4. Slow predictions → Check CPU usage

---

## Version History

- **v1.0** (Dec 5, 2025): Initial release
  - 90 models integrated
  - 3 prediction types
  - 15 disease classes
  - Modern web interface

---

## Technical Contact

**Repository**: https://github.com/gowthamalasakaniS574661/GDP-TEAM2  
**Branch**: merge/import-gdp-team2-into-main  
**Team**: GDP-TEAM2  

---

**Document Version**: 1.0  
**Last Updated**: December 5, 2025  
**Author**: GDP-TEAM2 Technical Team
