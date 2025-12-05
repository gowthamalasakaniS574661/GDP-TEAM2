# 🌿 Plant Disease Detection System - Project Documentation

## Project Information

**Project Name**: Plant Disease Detection System  
**Repository**: GDP-TEAM2  
**Owner**: gowthamalasakaniS574661  
**Branch**: merge/import-gdp-team2-into-main  
**Submission Date**: December 5, 2025  

---

## Executive Summary

This project implements a comprehensive AI-powered web application for detecting plant diseases using multiple machine learning approaches. The system supports three distinct prediction methods: text-based symptom analysis, quantitative feature analysis, and image-based diagnosis. With 90 trained models and 15 disease classifications across Tomato, Potato, and Pepper plants, the application provides accurate, real-time disease detection through an intuitive web interface.

---

## Project Objectives

### Primary Objectives
1. Develop a multi-modal plant disease detection system
2. Implement text, quantitative, and image-based prediction methods
3. Create a user-friendly web interface for farmers and agricultural professionals
4. Achieve high accuracy in disease classification across 15 disease categories
5. Provide confidence scores for predictions to ensure reliability

### Secondary Objectives
1. Support multiple ML algorithms for comparative analysis
2. Implement robust input validation and error handling
3. Ensure scalability for future model additions
4. Create comprehensive documentation for deployment

---

## System Architecture

### Technology Stack

#### Backend
- **Framework**: Flask 3.x (Python web framework)
- **Language**: Python 3.9+
- **ML Libraries**: 
  - scikit-learn 1.7.2 (Classical ML)
  - TensorFlow 2.13+ (Deep Learning)
  - joblib (Model serialization)

#### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Pillow & OpenCV**: Image preprocessing

#### Frontend
- **HTML5**: Structure and layout
- **CSS3**: Styling with gradients and responsive design
- **JavaScript (Vanilla)**: Asynchronous API calls and dynamic UI

### System Components

```
┌─────────────────────────────────────────────────┐
│           Web Browser (User Interface)          │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Flask Web Server (app.py)               │
│  ┌──────────────────────────────────────────┐  │
│  │    Route Handlers & API Endpoints        │  │
│  └──────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│           Prediction Engine                     │
│  ┌──────────────┬──────────────┬──────────────┐│
│  │ Text Models  │ Quant Models │ Image Models ││
│  │   (48)       │    (37)      │    (5)       ││
│  └──────────────┴──────────────┴──────────────┘│
└─────────────────────────────────────────────────┘
```

---

## Features & Capabilities

### 1. Text-Based Prediction (48 Models)

**Purpose**: Analyze written descriptions of plant symptoms to identify diseases

**Features**:
- Natural language processing of symptom descriptions
- TF-IDF vectorization for text feature extraction
- Multiple classifier algorithms:
  - Naive Bayes (Bernoulli, Complement, Multinomial)
  - Linear Support Vector Classifier (LinearSVC)
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Stochastic Gradient Descent (SGD)

**Input Validation**:
- Minimum 5 characters
- Vowel ratio check (≥15%)
- Consonant run detection (max 5 consecutive)
- Gibberish rejection with helpful error messages

**Example Inputs**:
- "leaves turning yellow with brown spots"
- "wilting plant with dark lesions on stems"
- "white powdery substance on tomato leaves"

### 2. Quantitative Prediction (37 Models)

**Purpose**: Process numerical measurements and environmental data

**Features**:
- Dynamic feature input based on selected model
- Support for various feature counts (adjustable per model)
- Multiple ML algorithms trained on numerical features
- Real-time feature count display

**Typical Features**:
- Temperature readings
- Humidity levels
- Soil pH
- Nutrient concentrations
- Growth measurements

**Input Format**: Numeric values (integers or decimals)

### 3. Image-Based Prediction (5 Models)

**Purpose**: Analyze photographs of plant leaves to identify diseases

**Features**:
- Deep learning models (CNN architectures)
- Support for multiple image formats (JPG, PNG)
- Drag-and-drop interface
- Image preview before prediction
- Automatic image preprocessing and resizing

**Models**:
1. Original CNN model (saved_model.pb)
2. Xception.keras (Transfer learning)
3. Feature Extractor (extractor.keras)
4. Custom Model M (m.keras)
5. General Model (model.keras)

**Specifications**:
- Input size: 224x224 or 256x256 RGB
- Maximum file size: 10MB
- Supported formats: JPG, PNG
- Output: Disease class with confidence score

---

## Disease Classifications

### Supported Plants & Diseases (15 Classes)

#### Pepper (Bell) - 2 Classes
1. **Bacterial Spot** - Bacterial infection causing dark lesions
2. **Healthy** - No disease detected

#### Potato - 3 Classes
3. **Early Blight** - Fungal disease with concentric ring patterns
4. **Late Blight** - Serious fungal disease causing rapid decay
5. **Healthy** - No disease detected

#### Tomato - 10 Classes
6. **Bacterial Spot** - Bacterial infection on leaves and fruit
7. **Early Blight** - Fungal disease with target-like lesions
8. **Late Blight** - Devastating fungal disease
9. **Leaf Mold** - Fungal infection on leaf undersides
10. **Septoria Leaf Spot** - Fungal disease with small circular spots
11. **Spider Mites (Two-spotted)** - Pest infestation damage
12. **Target Spot** - Fungal disease with bullseye patterns
13. **Yellow Leaf Curl Virus** - Viral disease causing leaf curling
14. **Tomato Mosaic Virus** - Viral disease causing mottled leaves
15. **Healthy** - No disease detected

---

## Model Performance & Statistics

### Overall Statistics
- **Total Models**: 90 trained models
- **Prediction Types**: 3 (Text, Quantitative, Image)
- **Disease Classes**: 15 across 3 plant types
- **Average Response Time**: <2 seconds per prediction
- **Confidence Threshold**: Configurable (default: varies by model)

### Model Distribution
- **Text Models**: 48 (53.3%)
- **Quantitative Models**: 37 (41.1%)
- **Image Models**: 5 (5.6%)

### Feature Selection Methods (Text Models)
- K-Best Mutual Information (500 features)
- Percentile Mutual Information (20%)
- SelectFromModel with ExtraTrees
- Variance Threshold
- Singular Value Decomposition (100 components)

---

## User Interface Design

### Navigation Flow
```
Home Page (/)
    │
    ├─→ Text Prediction (/text)
    │       └─→ Select Model → Enter Symptoms → Get Results
    │
    ├─→ Quantitative Prediction (/quant)
    │       └─→ Select Model → Enter Features → Get Results
    │
    └─→ Image Prediction (/image)
            └─→ Select Model → Upload Image → Get Results
```

### Design Principles
- **Minimalist**: Clean, uncluttered interface
- **Dark Theme**: Easy on eyes with gradient backgrounds
- **Responsive**: Works on desktop and tablet devices
- **Intuitive**: Clear navigation with back buttons
- **Feedback**: Loading states and error messages
- **Accessibility**: High contrast, readable fonts

### Color Scheme
- **Background**: Dark gradient (#03031a → #071124)
- **Accent Primary**: Cyan-green (#6ef0c3)
- **Accent Secondary**: Blue (#5a8bff)
- **Text**: Light blue-white (#e6f7ff)
- **Success**: Light cyan (#caffdf)
- **Error**: Coral red (#ff6b6b)

---

## API Documentation

### Endpoints

#### 1. GET `/`
**Description**: Home page with navigation cards  
**Returns**: HTML page

#### 2. GET `/text`
**Description**: Text prediction interface  
**Returns**: HTML page

#### 3. GET `/quant`
**Description**: Quantitative prediction interface  
**Returns**: HTML page

#### 4. GET `/image`
**Description**: Image prediction interface  
**Returns**: HTML page

#### 5. GET `/api/models`
**Description**: List all text-based models  
**Returns**: 
```json
{
  "models": [
    {
      "filename": "kbest_mi_500__LinearSVC.joblib",
      "size_bytes": 12345,
      "vectorizer": "tfidf.joblib"
    }
  ]
}
```

#### 6. POST `/api/predict`
**Description**: Text-based disease prediction  
**Request Body**:
```json
{
  "model_filename": "kbest_mi_500__LinearSVC.joblib",
  "text": "leaves turning yellow with brown spots"
}
```
**Response**:
```json
{
  "prediction": "Tomato_Early_blight",
  "confidence": 0.87,
  "used_model": "kbest_mi_500__LinearSVC.joblib",
  "vectorizer_used": "tfidf.joblib"
}
```

#### 7. GET `/api/quant_models`
**Description**: List all quantitative models  
**Returns**: JSON array of model objects with expected features

#### 8. POST `/api/predict_quant`
**Description**: Quantitative prediction  
**Request Body**:
```json
{
  "model": "quant_model_name.joblib",
  "features": [25.5, 60.3, 7.2, 15.8]
}
```

#### 9. GET `/api/image_models`
**Description**: List all image models  
**Returns**: JSON array with model names and supported classes

#### 10. POST `/api/predict_image`
**Description**: Image-based prediction  
**Request**: Multipart form data
- `image`: Image file (JPG/PNG)
- `model_name`: Selected model name

**Response**:
```json
{
  "prediction": "Tomato_Late_blight",
  "confidence": 0.92,
  "model": "Xception.keras",
  "prediction_index": 7
}
```

---

## Installation & Deployment

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

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

4. **Run Application**
```bash
python3 app.py
```

5. **Access in Browser**
```
http://localhost:5001
```

### Dependencies
- Flask>=3.0.0
- tensorflow>=2.13.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0
- Pillow>=10.0.0
- opencv-python>=4.8.0
- joblib>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0

---

## Project Structure

```
GDP-TEAM2/
│
├── final_submission/              # Submission folder
│   ├── PROJECT_DOCUMENTATION.md   # This file
│   └── technical/                 # Technical documents
│
├── app.py                         # Main Flask application (1231 lines)
├── requirements.txt               # Python dependencies
├── README.md                      # Setup and usage guide
│
├── templates/                     # HTML templates
│   ├── home.html                 # Landing page
│   ├── text.html                 # Text prediction UI
│   ├── quant.html                # Quantitative prediction UI
│   ├── image.html                # Image prediction UI
│   └── index.html                # Original template (legacy)
│
├── MLALGO/                        # Models directory
│   ├── text_model_artifacts/     # 48 text models + vectorizers
│   │   ├── tfidf.joblib
│   │   ├── label_encoder.joblib
│   │   ├── kbest_mi_500__*.joblib
│   │   ├── percentile_mi_20__*.joblib
│   │   ├── selectfrom_extratrees__*.joblib
│   │   ├── variance__*.joblib
│   │   └── svd_dense_100__*.joblib
│   │
│   ├── quant_model_artifacts/    # 37 quantitative models
│   │   └── [model files...]
│   │
│   ├── image_models/             # Original image model
│   │   └── model/
│   │       ├── saved_model.pb
│   │       ├── variables/
│   │       └── class_indices.json
│   │
│   └── image_models_all/         # Additional Keras models
│       ├── Xception.keras        (~88MB)
│       ├── extractor.keras       (~88MB)
│       ├── m.keras              (~299MB)
│       └── model.keras          (~270MB)
│
├── plant_disease_data.csv        # Training dataset
├── agroai_sept.py               # Legacy script
└── text_based_prediction.ipynb  # Jupyter notebook
```

---

## Key Achievements

### Technical Achievements
1. ✅ Integrated 90 trained models into single application
2. ✅ Implemented three distinct prediction methodologies
3. ✅ Created responsive, modern web interface
4. ✅ Achieved real-time predictions (<2 seconds)
5. ✅ Implemented robust input validation
6. ✅ Added automatic gibberish detection for text inputs
7. ✅ Developed dynamic feature input system for quantitative models
8. ✅ Implemented drag-and-drop image upload
9. ✅ Created comprehensive error handling
10. ✅ Built RESTful API architecture

### User Experience Achievements
1. ✅ Clean, intuitive navigation system
2. ✅ Visual feedback with loading states
3. ✅ Confidence score visualization
4. ✅ Helpful error messages
5. ✅ Model information display
6. ✅ Image preview functionality
7. ✅ Responsive design for multiple devices

### Documentation Achievements
1. ✅ Comprehensive README with installation guide
2. ✅ Step-by-step troubleshooting section
3. ✅ API endpoint documentation
4. ✅ Project structure documentation
5. ✅ Complete project documentation (this file)

---

## Challenges & Solutions

### Challenge 1: Model Integration
**Problem**: Different model types required different loading mechanisms
**Solution**: Created adapter classes and unified loading functions

### Challenge 2: Directory Structure Mismatches
**Problem**: Models stored in unexpected directory paths
**Solution**: Implemented fallback search paths and dynamic directory detection

### Challenge 3: Label Consistency
**Problem**: New image models returned numeric predictions instead of disease names
**Solution**: Implemented shared label encoder with fallback to class_indices.json

### Challenge 4: Input Validation
**Problem**: Users entering random text causing poor predictions
**Solution**: Developed sophisticated gibberish detection with multiple heuristics

### Challenge 5: Feature Count Variability
**Problem**: Quantitative models expecting different numbers of features
**Solution**: Dynamic form generation based on selected model's expected_features

### Challenge 6: Large Model Files
**Problem**: TensorFlow models (~745MB total) causing slow load times
**Solution**: Lazy loading of models only when needed

### Challenge 7: Port Conflicts
**Problem**: Development server conflicts when restarting
**Solution**: Added port cleanup instructions and process management

---

## Testing & Validation

### Test Cases Executed

#### Text Prediction Tests
- ✅ Valid symptom descriptions → Correct predictions
- ✅ Gibberish input → Proper rejection with error message
- ✅ Short input (<5 chars) → Validation error
- ✅ All 48 models → Successful loading and prediction
- ✅ Confidence scores → Within [0, 1] range

#### Quantitative Prediction Tests
- ✅ Correct feature count → Successful prediction
- ✅ Wrong feature count → Proper error message
- ✅ Non-numeric input → Validation error
- ✅ All 37 models → Successful loading
- ✅ Dual field support (`model` and `model_name`) → Both work

#### Image Prediction Tests
- ✅ Valid JPG/PNG images → Correct predictions
- ✅ Large images (>10MB) → Size validation
- ✅ Invalid formats → Proper error handling
- ✅ All 5 models → Disease name output (not numbers)
- ✅ Drag-and-drop → Functional
- ✅ Image preview → Displays correctly

#### Integration Tests
- ✅ Navigation flow → All pages accessible
- ✅ Back buttons → Return to home
- ✅ API endpoints → All 10 endpoints functional
- ✅ Error handling → Graceful degradation
- ✅ Loading states → Proper UI feedback

---

## Performance Metrics

### Response Times (Average)
- Text Prediction: 0.5-1.5 seconds
- Quantitative Prediction: 0.3-0.8 seconds
- Image Prediction: 1.0-2.5 seconds
- Model Loading (startup): 5-10 seconds
- Page Load Time: <1 second

### Resource Usage
- Memory (idle): ~200MB
- Memory (with image model loaded): ~800MB
- CPU (during prediction): 30-60%
- Disk Space: ~2GB (with all models)

### Accuracy Metrics
- Overall system accuracy: Varies by model and disease type
- Text models: High accuracy for well-described symptoms
- Quantitative models: Dependent on feature quality
- Image models: Strong performance on clear images

---

## Future Enhancements

### Planned Features
1. **Mobile Application**: React Native or Flutter app
2. **Real-time Camera**: Live image capture and analysis
3. **Treatment Recommendations**: Suggest remedies per disease
4. **Multi-language Support**: Internationalization
5. **User Authentication**: Personal accounts and history
6. **Batch Processing**: Multiple image analysis
7. **Model Comparison**: Side-by-side model performance
8. **Export Reports**: PDF generation of results
9. **Offline Mode**: Local model caching
10. **Progressive Web App**: Installable web application

### Technical Improvements
1. Model compression for faster loading
2. Database integration for prediction history
3. Caching mechanism for frequent predictions
4. WebSocket support for real-time updates
5. Docker containerization
6. Kubernetes deployment
7. Load balancing for multiple instances
8. Model versioning and A/B testing
9. Enhanced security (HTTPS, input sanitization)
10. Performance monitoring and analytics

---

## Conclusion

The Plant Disease Detection System successfully demonstrates a comprehensive approach to agricultural disease diagnosis using multiple machine learning methodologies. With 90 trained models, 15 disease classifications, and an intuitive web interface, the system provides a practical tool for farmers, agricultural professionals, and researchers.

The project showcases:
- **Technical Excellence**: Integration of classical ML and deep learning
- **User-Centric Design**: Intuitive interface with robust validation
- **Scalability**: Architecture supports easy model additions
- **Reliability**: Comprehensive error handling and validation
- **Documentation**: Extensive guides for deployment and usage

This system represents a significant step toward accessible AI-powered agricultural solutions, with potential for real-world impact in crop disease management.

---

## References & Resources

### Datasets
- Plant Village Dataset
- Custom agricultural disease datasets

### Frameworks & Libraries
- Flask: https://flask.palletsprojects.com/
- TensorFlow: https://www.tensorflow.org/
- scikit-learn: https://scikit-learn.org/
- Keras: https://keras.io/

### Research Papers
- Deep learning for plant disease detection
- Transfer learning in agricultural applications
- Text mining for symptom analysis

---

## Contact Information

**Repository**: https://github.com/gowthamalasakaniS574661/GDP-TEAM2  
**Branch**: merge/import-gdp-team2-into-main  
**Project Team**: GDP-TEAM2  

---

**Document Version**: 1.0  
**Last Updated**: December 5, 2025  
**Total Pages**: Comprehensive project documentation  

---

*This project is submitted as part of the Graduate Directed Project (GDP) coursework.*
