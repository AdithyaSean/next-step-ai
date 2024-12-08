# Next Step AI Career Guidance API

## Overview
RESTful API for the Next Step AI career guidance system, built with FastAPI.

## Features
- User authentication with JWT
- Student profile management
- Career guidance predictions
- Skills and career paths tracking
- Comprehensive data validation
- Async database operations
- CORS support
- Environment-based configuration

## API Endpoints

### Authentication
- `POST /api/v1/token` - Get access token
- `POST /api/v1/users/` - Create new user
- `GET /api/v1/users/me/` - Get current user

### Profile Management
- `POST /api/v1/profile/` - Create/update profile
- `GET /api/v1/profile/` - Get profile

### Career Guidance
- `POST /api/v1/guidance/` - Get career recommendations
- `GET /api/v1/paths/` - List available career paths
- `GET /api/v1/skills/` - List tracked skills

### System
- `GET /api/v1/health` - Health check
- `GET /` - API information

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configurations
```

4. Run development server:
```bash
uvicorn src.api.main:app --reload
```

## API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Data Models

### Student Profile
```python
StudentProfile:
    - student_id: str
    - education_level: EducationLevel
    - ol_results: OLResults
    - al_results: Optional[ALResults]
    - university_data: Optional[UniversityData]
    - skills_assessment: Dict[str, int]
    - interests: List[str]
    - career_preferences: Optional[CareerPreferences]
    - constraints: Optional[Dict]
```

### Career Guidance Response
```python
CareerGuidanceResponse:
    - student_id: str
    - timestamp: str
    - top_career_paths: List[CareerPathScore]
    - skill_gaps: Dict[str, List[str]]
    - recommended_courses: List[str]
    - growth_opportunities: List[str]
```

## Security
- JWT authentication
- Password hashing with bcrypt
- CORS protection
- Environment-based secrets

## Database
- Async SQLAlchemy with SQLite
- Migrations support
- Connection pooling

## Error Handling
- Comprehensive validation
- Detailed error messages
- HTTP status codes
- Exception middleware

## Development
1. Run tests:
```bash
pytest
```

2. Format code:
```bash
black src/
```

3. Check types:
```bash
mypy src/
```

## Production Deployment
1. Update `.env` with production settings
2. Use production-grade database
3. Configure CORS properly
4. Set up logging
5. Use HTTPS
6. Set up monitoring
