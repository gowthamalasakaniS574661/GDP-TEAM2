# Plant Disease Detection System - GDP-TEAM2

## Project Submission

This repository contains the complete implementation of the Plant Disease Detection System.

### For Project Evaluation

All project files are organized in the final_submission folder as per submission requirements.

Please navigate to:
```
final_submission/
```

---

## Folder Structure

```
MLALGO/
│
└── final_submission/              [MAIN SUBMISSION FOLDER]
    │
    ├── PROJECT_DOCUMENTATION.md   - Complete project documentation
    ├── README.md                  - Setup and usage guide  
    ├── SUBMISSION_CHECKLIST.md    - Verification checklist
    ├── SUBMISSION_COMPLETE.md     - Submission confirmation
    │
    └── technical/                 [ALL TECHNICAL FILES]
        ├── app.py                 - Main Flask application
        ├── requirements.txt       - Dependencies
        ├── TECHNICAL_DOCUMENTATION.md  - Technical details
        │
        ├── MLALGO/               - All ML models (90 models)
        │   ├── text_model_artifacts/     (48 models)
        │   ├── quant_model_artifacts/    (37 models)
        │   ├── image_models/             (image models)
        │   └── image_models_all/         (Keras models)
        │
        ├── templates/            - Web interface (HTML files)
        ├── plant_disease_data.csv - Dataset
        ├── text_based_prediction.ipynb  - Jupyter notebook
        └── [other implementation files]
```

---

## Quick Start

### Option 1: Navigate to Submission Folder
```bash
cd final_submission/
# Read PROJECT_DOCUMENTATION.md for complete overview
# Read technical/TECHNICAL_DOCUMENTATION.md for implementation details
```

### Option 2: Run the Application
```bash
cd final_submission/technical/
pip3 install -r requirements.txt
python3 app.py
# Visit http://localhost:5001
```

---

## Documentation

All documentation is located in the final_submission folder:

1. PROJECT_DOCUMENTATION.md - Complete project overview
2. README.md - Detailed setup and usage instructions
3. technical/TECHNICAL_DOCUMENTATION.md - Implementation details and API specs

---

## Submission Compliance

This project follows the required submission structure:

- (a) Created folder final_submission
- (b) Created folder technical inside final_submission
- (c) Project documentation in final_submission folder
- (d) All technical files and implementation in technical folder

---

## Repository Information

Repository: https://github.com/44-691-SU25/MLALGO  
Main Branch: main  

---

## Project Summary

- 90 ML Models (48 text, 37 quantitative, 5 image)
- 15 Disease Classifications (Tomato, Potato, Pepper)
- 3 Prediction Methods (Text, Quantitative, Image-based)
- Modern Web Interface with REST API
- Complete Documentation (1,800+ lines)

---

For complete project details, please see the final_submission folder.

Submitted by: GDP-TEAM2  
Date: December 5, 2025
