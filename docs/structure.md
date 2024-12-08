# Project Structure Documentation

## Current Structure
```
next-step-ai/
├── data/                      # Data storage and processing
│   ├── raw/                  # Raw, immutable data
│   └── processed/            # Cleaned and processed data
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
│   │   ├── preprocessor.py # Data preprocessing
│   │   └── validator.py   # Data validation
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
- [2024-12-08] Added FastAPI Implementation
  - Created complete API structure
  - Added authentication system
  - Implemented data validation
  - Added database operations
  - Created API documentation

- [2024-12-08] Updated Data Schema
  - Added university-level data points
  - Enhanced career preferences
  - Added technical competencies
  - Improved validation rules

## Key Components

### API Layer (`src/api/`)
- `main.py`: FastAPI application setup
- `models.py`: Pydantic models for request/response
- `routes.py`: API endpoints implementation
- `auth.py`: JWT authentication
- `database.py`: Async database operations

### Data Processing (`src/data/`)
- `validator.py`: Data validation logic
- `preprocessor.py`: Data preprocessing utilities

### ML Models (`src/models/`)
- `career_predictor.py`: Career prediction implementation

### Documentation (`docs/`)
- `data_schema.md`: Comprehensive data schema
- `structure.md`: Project structure (this file)
- `roadmap/ROADMAP.md`: Development roadmap

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

## API Endpoints

### Authentication
- POST /api/v1/token
- POST /api/v1/users/
- GET /api/v1/users/me/

### Profile Management
- POST /api/v1/profile/
- GET /api/v1/profile/

### Career Guidance
- POST /api/v1/guidance/
- GET /api/v1/paths/
- GET /api/v1/skills/

### System
- GET /api/v1/health
- GET /

## Database Schema
- Users
- Student Profiles
- Academic Records
- Career Preferences
- Guidance History

## Future Additions
- API tests
- Database migrations
- Monitoring system
- Mobile optimization
- Flutter integration

## Development Guidelines
1. Follow FastAPI best practices
2. Use async/await for database operations
3. Validate all input data
4. Document all endpoints
5. Write tests for new features
6. Keep dependencies minimal
7. Maintain security best practices

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
