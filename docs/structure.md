# Project Structure Documentation

## Current Structure
```
next-step-ai/
├── data/                      # Data storage and processing
│   ├── raw/                  # Raw generated datasets
│   ├── processed/            # Cleaned and processed data
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
│   └── generate_dataset.py  # Dataset generation script
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
│   │   └── career_predictor.py  # Main prediction model
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
- [2024-12-08] Added Data Generation System
  - Created dataset generator with Sri Lankan education patterns
  - Added configuration for realistic distributions
  - Implemented compatibility checker with API models
  - Added sample data generation script

- [2024-12-08] Added FastAPI Implementation
  - Created complete API structure
  - Added authentication system
  - Implemented data validation
  - Added database operations

## Key Components

### Data Generation (`src/data/generators/`)
- `config.py`: Configuration for realistic data generation
- `dataset_generator.py`: Main data generation logic
- `compatibility_check.py`: API compatibility verification

### API Layer (`src/api/`)
- `models.py`: Pydantic models for request/response
- `routes.py`: API endpoints implementation
- `auth.py`: JWT authentication
- `database.py`: Async database operations

### Data Processing (`src/data/`)
- `validator.py`: Data validation logic
- `preprocessor.py`: Data preprocessing utilities

### ML Models (`src/models/`)
- `career_predictor.py`: Career prediction implementation

## Data Flow
1. Data Generation → Validation → Storage
2. API Request → Validation → Processing → Response
3. Model Training → Evaluation → Deployment

## Dependencies
Current dependencies:
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

## Generated Data Structure
1. Student Profiles (JSON/CSV)
   - Academic records
   - Skills assessment
   - Career preferences
   - Constraints

2. Sample Data
   - Test cases
   - Validation examples
   - API compatibility checks

## Development Guidelines
1. Follow FastAPI best practices
2. Use async/await for database operations
3. Validate all input data
4. Document all endpoints
5. Write tests for new features
6. Keep dependencies minimal
7. Maintain security best practices

## Data Generation Guidelines
1. Follow Sri Lankan education patterns
2. Use realistic distributions
3. Maintain data relationships
4. Validate against API models
5. Generate reproducible datasets

## Testing Strategy
1. Unit tests for generators
2. Validation tests for API models
3. Integration tests for data flow
4. Performance tests for model
5. API endpoint tests

## Security Measures
1. JWT authentication
2. Password hashing
3. Environment variables
4. CORS protection
5. Input validation
6. Rate limiting (to be added)

## Notes
- Keep raw data immutable
- Use processed/ for modified data
- Update documentation when making changes
- Follow API versioning
- Maintain test coverage
- Validate generated data
