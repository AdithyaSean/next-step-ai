# Project Structure Documentation

## Current Structure
```
next-step-ai/
├── data/                      # Data storage and processing
│   ├── raw/                  # Raw generated datasets
│   │   └── student_profiles.json
│   ├── processed/            # Cleaned and processed data
│   │   └── processed_profiles.csv
│   ├── models/              # Saved models and preprocessors
│   │   ├── preprocessor/    # Saved preprocessor state
│   │   └── career_predictor/  # Saved model state
│   └── samples/             # Sample data for testing
│
├── docs/                     # Documentation
│   ├── roadmap/             # Project roadmap and milestones
│   │   └── ROADMAP.md       # Detailed development roadmap
│   ├── structure.md         # Project structure documentation
│   └── data_schema.md       # Data collection schema
│
├── models/                   # Trained models
│   └── saved/               # Saved model files
│
├── scripts/                  # Utility scripts
│   ├── generate_dataset.py  # Dataset generation script
│   ├── process_data.py      # Data processing script
│   └── train_model.py       # Model training script
│
├── src/                     # Source code
│   ├── api/                # API implementation
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI application
│   │   ├── models.py      # Pydantic models
│   │   ├── routes.py      # API endpoints
│   │   ├── auth.py        # Authentication
│   │   ├── database.py    # Database operations
│   │   └── README.md      # API documentation
│   │
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   ├── generators/    # Data generation
│   │   │   ├── __init__.py
│   │   │   ├── config.py  # Generation configuration
│   │   │   └── dataset_generator.py
│   │   ├── preprocessor.py # Data preprocessing
│   │   ├── validator.py   # Data validation
│   │   └── compatibility_check.py  # API compatibility checker
│   │
│   ├── models/            # ML models
│   │   ├── __init__.py
│   │   └── career_predictor.py  # Career prediction model
│   │
│   └── utils/             # Utility functions
│       └── __init__.py
│
├── tests/                   # Test files
│   ├── __init__.py
│   ├── data/              # Data processing tests
│   │   ├── __init__.py
│   │   ├── test_generator.py
│   │   └── test_validator.py
│   └── models/             # Model tests
│       └── __init__.py
│
├── .env.example            # Environment variables template
├── .gitignore             # Git ignore file
├── README.md              # Project overview
├── requirements.txt       # Project dependencies
└── setup.sh              # Environment setup script
```

## Recent Changes
- [2024-12-08] Added Model Development
  - Created LightGBM-based career predictor
  - Added model training pipeline
  - Implemented feature importance analysis
  - Added model persistence

- [2024-12-08] Enhanced Data Pipeline
  - Added data preprocessing system
  - Created data validation
  - Implemented data compatibility checker
  - Added sample data generation

## Key Components

### Data Generation (`src/data/generators/`)
- `config.py`: Configuration for realistic data generation
- `dataset_generator.py`: Main data generation logic
- `compatibility_check.py`: API compatibility verification

### Data Processing (`src/data/`)
- `preprocessor.py`: Data preprocessing pipeline
- `validator.py`: Data validation logic

### Model Implementation (`src/models/`)
- `career_predictor.py`: LightGBM-based prediction model
  - Training pipeline
  - Feature importance analysis
  - Model persistence
  - Performance metrics

### Scripts
- `generate_dataset.py`: Generate sample data
- `process_data.py`: Preprocess and validate data
- `train_model.py`: Train and evaluate model

## Data Flow
1. Data Generation → Validation → Storage
2. Raw Data → Preprocessing → Training Data
3. Training Data → Model Training → Evaluation
4. Model → Persistence → Deployment

## Dependencies
```
Core ML:
- lightgbm>=4.1.0
- numpy>=1.24.3
- pandas>=2.0.3
- scikit-learn>=1.3.0
- joblib>=1.3.1

API:
- fastapi>=0.109.0
- uvicorn>=0.27.0
- pydantic>=2.5.0
- python-multipart>=0.0.6
- python-jose[cryptography]>=3.3.0
- passlib[bcrypt]>=1.7.4
- python-dotenv>=1.0.0
```

## Development Guidelines
1. Follow modular design
2. Validate all data transformations
3. Document all components
4. Write comprehensive tests
5. Monitor model performance
6. Keep code maintainable

## Data Generation Guidelines
1. Follow Sri Lankan education patterns
2. Use realistic distributions
3. Maintain data relationships
4. Validate against API models
5. Generate reproducible datasets

## Model Development Guidelines
1. Use LightGBM for efficiency
2. Monitor feature importance
3. Validate predictions
4. Maintain interpretability
5. Track performance metrics

## Testing Strategy
1. Unit tests for components
2. Integration tests for pipeline
3. Model performance tests
4. API endpoint tests
5. Data validation tests

## Security Measures
1. JWT authentication
2. Password hashing
3. Environment variables
4. CORS protection
5. Input validation

## Notes
- Keep raw data immutable
- Monitor model metrics
- Update documentation
- Follow versioning
- Maintain test coverage
