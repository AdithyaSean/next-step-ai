from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import List
from datetime import timedelta

from .models import (
    StudentProfile, CareerGuidanceResponse, User, 
    UserCreate, Token, CareerPathScore
)
from .auth import (
    authenticate_user, create_access_token, 
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .database import get_db
from ..models.career_predictor import CareerPredictor

router = APIRouter()
predictor = CareerPredictor()

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """Login endpoint to get access token."""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@router.post("/users/", response_model=User)
async def create_user(user: UserCreate, db=Depends(get_db)) -> User:
    """Create new user."""
    db_user = await get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    return await create_new_user(db=db, user=user)

@router.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user profile."""
    return current_user

@router.post("/profile/", response_model=StudentProfile)
async def create_profile(
    profile: StudentProfile,
    current_user: User = Depends(get_current_active_user)
) -> StudentProfile:
    """Create or update student profile."""
    # Add validation logic here
    return await update_user_profile(current_user.id, profile)

@router.get("/profile/", response_model=StudentProfile)
async def get_profile(
    current_user: User = Depends(get_current_active_user)
) -> StudentProfile:
    """Get student profile."""
    profile = await get_user_profile(current_user.id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Profile not found"
        )
    return profile

@router.post("/guidance/", response_model=CareerGuidanceResponse)
async def get_career_guidance(
    profile: StudentProfile,
    current_user: User = Depends(get_current_active_user)
) -> CareerGuidanceResponse:
    """Get career guidance recommendations."""
    try:
        # Get predictions from the model
        predictions = predictor.predict(profile)
        
        # Convert predictions to response format
        career_paths = [
            CareerPathScore(
                path_name=pred["path"],
                score=pred["score"],
                confidence=pred["confidence"],
                reasons=pred["reasons"],
                requirements=pred["requirements"],
                next_steps=pred["next_steps"]
            )
            for pred in predictions["career_paths"]
        ]
        
        return CareerGuidanceResponse(
            student_id=profile.student_id,
            timestamp=predictions["timestamp"],
            top_career_paths=career_paths,
            skill_gaps=predictions["skill_gaps"],
            recommended_courses=predictions["recommended_courses"],
            growth_opportunities=predictions["growth_opportunities"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating career guidance: {str(e)}"
        )

@router.get("/paths/", response_model=List[str])
async def get_available_paths() -> List[str]:
    """Get list of available career paths."""
    return predictor.get_available_paths()

@router.get("/skills/", response_model=List[str])
async def get_required_skills() -> List[str]:
    """Get list of tracked skills."""
    return predictor.get_tracked_skills()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_version": predictor.version}
