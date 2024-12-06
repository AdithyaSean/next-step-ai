# Next Step - Career Guidance System

A machine learning system to help Sri Lankan students make informed career decisions based on their academic performance, interests, and skills.

## Overview
This repository is dedicated to exploring and identifying the best AI model and datasets for building Next Step, a career guidance application. The final product will be a multiplatform application developed using Flutter for our Innovation and Entrepreneurship module and NIBM HND in Software Engineering's capstone project. The app aims to assist Sri Lankan students in making informed educational and career decisions.

## Objective
The primary goal of this project is to experiment with various AI models and datasets to determine the most effective solution for:

1. Predicting suitable academic and career paths based on a student's exam performance, academic progress, and interests.
2. Providing multiple recommendations tailored to a student's current academic level.
3. Ensuring the model can self-evolve by incorporating user-provided data.

## Features
- **Prediction Models:** Suggest educational streams or career paths for OL, AL, and campus-level students.
- **Dynamic Recommendations:** Generate personalized advice based on user input.
- **Dataset Exploration:** Leverage datasets that include academic performance, career outcomes, and interests.
- **AI Model Comparison:** Evaluate the effectiveness of different machine learning models (e.g., Decision Trees, Neural Networks, Gradient Boosted Models).

## Dataset Requirements
We aim to use datasets that include:
- Academic marks and progress across OLs, ALs, and campus.
- Career outcomes (e.g., graduation status, job acquisition).
- Initial career paths and long-term trajectories.
- User interests and extracurricular achievements.

### Sources
- Publicly available datasets on education and employment.
- Synthetic datasets created for simulating Sri Lankan students' academic and career journeys.

## Development Plan
This exploratory project will proceed as follows:

1. **Dataset Collection:** Identify and preprocess datasets that match our requirements.
2. **Model Training:** Train various machine learning models to predict educational and career paths.
3. **Evaluation Metrics:** Use metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
4. **Iteration:** Fine-tune models and preprocess data to improve performance.
5. **Integration Preparation:** Ensure the chosen model is ready for integration with the Flutter frontend.

## Quick Start

1. Set up your environment:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

2. Generate sample data for testing:
```bash
python scripts/generate_sample_data.py
```
This will create sample student data in `data/raw/sample_data.csv`

3. Train and test the model:
```bash
# Prepare the data
python scripts/prepare_data.py --input data/raw/sample_data.csv

# Initial model training
python scripts/train_model.py --data data/processed/train.csv

# Test predictions or train incrementally
python scripts/test_predictions.py  # For testing only
python scripts/test_predictions.py --train path/to/new_data.csv  # For incremental training
```

The model supports two modes of operation:
- **Testing Mode**: Evaluates model performance without modifying the trained model
- **Training Mode**: Updates the existing model with new data to improve predictions

### Important Notes:
- New training data must follow the same format as the original dataset
- The model and preprocessor states are saved in `models/` directory
- Training mode will preserve previous learning while incorporating new patterns
- Success rates and confidence scores are shown for each prediction

## Model Training and Testing

1. Regular testing (won't update model):

## Model Architecture

Our career guidance system uses a three-stage prediction approach:

### Stage 1: A/L Stream Predictor
- **Input**: O/L subject results
- **Model**: XGBoost Classifier
- **Output**: Recommended A/L streams with confidence scores
- **Features**:
  - Core subject performance (Mathematics, Science, English, etc.)
  - Stream-specific eligibility checks
  - Confidence scores for each stream

### Stage 2: University Program Predictor
- **Input**: O/L + A/L results, Z-score
- **Model**: LightGBM Classifier
- **Output**: Recommended university programs
- **Features**:
  - Program-specific eligibility checks
  - Z-score requirements
  - Subject requirements validation
  - Success probability scores

### Stage 3: Career Path Predictor
- **Input**: Complete educational profile
- **Model**: Ensemble (XGBoost + Random Forest)
- **Output**: Career path recommendations
- **Features**:
  - Weighted ensemble predictions
  - Career-specific requirements
  - Required skills analysis
  - Industry alignment scoring

## Input Data Format

The system accepts student profiles with varying levels of detail:

### Example Input Formats

#### 1. O/L Student Profile
```python
{
    # Core O/L Subjects (Required)
    "ol_mathematics": 85,  # Mathematics
    "ol_science": 78,     # Science
    "ol_english": 82,     # English

    # Additional Information
    "interests": ["Science", "Technology", "Mathematics"],
    "skills": ["Problem Solving", "Analytical Thinking"],
    "extracurricular": ["Science Society", "Mathematics Olympiad"]
}
```

#### 2. A/L Student Profile
```python
{
    # Core O/L Results (Required)
    "ol_mathematics": 85,
    "ol_science": 78,
    "ol_english": 82,

    # A/L Stream Information
    "al_stream": "Physical Science",  # One of: "Physical Science", "Biological Science", "Commerce", "Arts"
    
    # Stream-specific subjects
    # For Physical Science Stream
    "al_physics": 75,
    "al_chemistry": 82,
    "al_mathematics": 88,  # Combined Mathematics
    "al_zscore": 1.8,

    # Additional Information
    "interests": ["Engineering", "Technology"],
    "skills": ["Mathematics", "Physics", "Programming"],
    "extracurricular": ["Robotics Club", "Science Exhibition"]
}
```

#### 3. Alternative A/L Stream Examples

##### Biological Science Stream
```python
{
    "al_stream": "Biological Science",
    "al_biology": 85,
    "al_chemistry": 78,
    "al_physics": 82,
    "al_zscore": 1.9,
    "interests": ["Medicine", "Research"],
    "skills": ["Laboratory Work", "Analysis"]
}
```

##### Commerce Stream
```python
{
    "al_stream": "Commerce",
    "al_accounting": 88,
    "al_business_studies": 85,
    "al_economics": 82,
    "al_zscore": 1.7,
    "interests": ["Business", "Finance"],
    "skills": ["Financial Analysis", "Management"]
}
```

##### Arts Stream
```python
{
    "al_stream": "Arts",
    "al_zscore": 1.6,
    "interests": ["Languages", "Social Sciences", "History"],
    "skills": ["Research", "Critical Thinking", "Writing"],
    "subjects": ["Political Science", "Geography", "Languages"]  # Main subjects taken
}
```

### Notes on Sri Lankan Education System

1. **O/L (Ordinary Level)**
   - Core subjects focused on Mathematics, Science, and English
   - Grades typically range from 0-100
   - Passing grade is typically 40

2. **A/L (Advanced Level)**
   - Four main streams: Physical Science, Biological Science, Commerce, and Arts
   - Z-score calculation considers:
     - Raw marks in each subject
     - Island-wide performance
     - District quota system

3. **Stream-Specific Requirements**
   - Physical Science: Mathematics, Physics, Chemistry
   - Biological Science: Biology, Physics, Chemistry
   - Commerce: Business Studies, Accounting, Economics
   - Arts: Flexible combination of subjects (e.g., Languages, Geography, Political Science)

4. **University Admission**
   - Based on Z-score ranking
   - Stream-specific course eligibility

## Output Format

The system provides stage-appropriate recommendations:

1. **A/L Stream Recommendations**:
```python
{
    "recommendations": [
        {
            "stream": "Science",
            "confidence": 0.85,
            "eligible": true,
            "requirements_met": {
                "ol_mathematics": true,
                "ol_science": true
            }
        },
        # ... other streams ...
    ]
}
```

2. **University Program Recommendations**:
```python
{
    "recommendations": [
        {
            "program": "Computer Science",
            "success_probability": 0.78,
            "eligible": true,
            "requirements_status": {
                "stream_match": true,
                "zscore_met": true,
                "subject_requirements": {
                    "combined_mathematics": {
                        "required": 65,
                        "actual": 88,
                        "met": true
                    }
                }
            }
        },
        # ... other programs ...
    ]
}
```

3. **Career Path Recommendations**:
```python
{
    "recommendations": [
        {
            "career": "Software Engineer",
            "confidence": 0.82,
            "requirements": {
                "education_level": "University",
                "preferred_programs": ["Computer Science", "Engineering"],
                "min_gpa": 2.75
            },
            "skills_needed": [
                "Programming",
                "Problem Solving",
                "Software Development",
                "Database Management"
            ]
        },
        # ... other careers ...
    ]
}
```

## Current Status

1. **Completed**:
   - Data generation pipeline
   - Three-stage model architecture
   - Base model implementations
   - Configuration system
   - Preprocessing pipeline

2. **In Progress**:
   - Model training pipeline
   - Evaluation metrics
   - Hyperparameter tuning
   - API layer development

3. **Upcoming**:
   - Test suite implementation
   - Model deployment strategy
   - Integration with Flutter frontend
   - User feedback loop

## Project Structure

Next Step AI/
├── data/                          # Data storage
│   ├── processed/                 # Processed datasets
│   └── raw/                      # Raw data files
│       └── sample_data.csv       # Generated sample data
│
├── docs/                         # Documentation
│   └── model_comparison.md       # Model evaluation and comparison
│
├── notebooks/                    # Jupyter notebooks
│   └── 00_quickstart_guide.ipynb # Getting started guide
│
├── scripts/                      # Utility scripts
│   ├── evaluate_model.py         # Model evaluation
│   ├── generate_sample_data.py   # Sample data generation
│   ├── predict_cli.py           # CLI prediction interface
│   ├── prepare_data.py          # Data preparation pipeline
│   ├── train_model.py           # Model training
│   └── tune_hyperparameters.py  # Hyperparameter optimization
│
├── src/                         # Source code
│   ├── config/                  # Configuration
│   │   ├── config.py           # Configuration management
│   │   ├── config.yaml         # General configuration
│   │   └── model_config.yaml   # Model-specific settings
│   │
│   ├── data/                    # Data processing
│   │   └── preprocessing.py    # Data preprocessing utilities
│   │
│   ├── models/                  # Model implementations
│   │   ├── base_model.py       # Abstract base class
│   │   ├── al_stream_predictor.py    # A/L stream prediction (Stage 1)
│   │   ├── university_predictor.py    # University program prediction (Stage 2)
│   │   ├── career_path_predictor.py   # Career path prediction (Stage 3)
│   │   ├── career_guidance_system.py  # Unified prediction interface
│   │   └── preprocessing.py     # Model-specific preprocessing
│   │
│   └── utils/                   # Utilities
│       └── logger.py           # Logging configuration
│
├── Makefile                     # Build automation
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies

## Roadmap
- **Phase 1:** Explore available datasets and preprocess them.
- **Phase 2:** Experiment with baseline models.
- **Phase 3:** Fine-tune the best-performing model.
- **Phase 4:** Prepare the model for integration into the Flutter application.

## Contributing
We welcome contributions! If you'd like to collaborate, please fork the repository, create a feature branch, and submit a pull request. Ensure your changes pass existing tests and include new tests where applicable.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For inquiries, reach out to:
- **Name:** Adithya Ekanayaka
- **Email:** adithyasean@gmail.com
- **LinkedIn:** (https://www.linkedin.com/in/adithyasean)

---
We look forward to building an innovative and impactful solution together!
