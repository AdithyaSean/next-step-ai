# NEXT STEP: Career Guidance AI Model

## Overview
This repository contains the AI model component for the Next Step career guidance application. The model uses a LightGBM for multi-label classification to predict suitable career paths based on student academic performance and interests.

## Features
- **Multi-label Career Path Prediction:** Predicts multiple suitable career paths simultaneously
- **Comprehensive Feature Engineering:** Handles various academic inputs (O/L results, A/L results, interests)
- **Education Level Awareness:** Adapts predictions based on student's current education level
- **Binary Career Path Encoding:** Efficient encoding of career paths for multi-label classification
- **Robust Data Preprocessing:** Handles missing values and categorical variables appropriately

## Technical Stack
- **Python 3.12+**
- **scikit-learn:** LightGBM for multi-label classification
- **pandas & numpy:** Data processing and numerical operations
- **tensorflow:** Model serialization

## Model Architecture
- **Type:** LightGBM Classifier with MultiOutputClassifier
- **Features:** 49 input features including:
  - O/L results (mathematics, science, english, etc.)
  - A/L results (if applicable)
  - Stream information
  - Z-score (if applicable)
  - Interests (encoded as binary features)
  - Other academic indicators
- **Output:**
    **5 probability percentages indicators for top 5**
    - AL streams if the student level is OL
    - Courses if the student level is AL
    - Career paths if the student level is Higher Education

## Current Performance
Based on synthetic data:
- Average Accuracy: >99% across all career paths
- Precision & Recall: >0.94 for all classes
- F1-Score: >0.94 for all classes

Note: These metrics are based on synthetic data and may not reflect real-world performance.

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher

### Setup
1. Run the setup script:
   ```bash
   ./setup.sh # for Linux/macOS
   .\setup.ps1 # for Windows
   ```

### Training the Model
```bash
# Generate synthetic dataset
python -m generate

# Preprocess the data
python -m preprocess

# Train the model
python -m train
```

## 📁 Project Structure
```
next-step-ai/
├── data/
│   ├── processed/            # Preprocessed datasets
│   └── raw/                  # Raw datasets
├── models/
│   ├── saved/                # Trained model files
│   └── converted/            # Converted model files
└── src/
    ├── config/               # Data Configuration
    │   ├── config.py
    ├── generators/           # Dataset generation scripts
    │   ├── generator.py
    ├── train/                # Dataset preprocessing scripts
    │   ├── preprocess.py
    ├── train/                # Train scripts
        └── train.py
```

## 📊 Future Improvements
1. Collect real student data for training
2. Implement cross-validation for better evaluation
3. Experiment with different model architectures
4. Add confidence scores for predictions
5. Implement model versioning and monitoring

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
