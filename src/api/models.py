from typing import List, Optional, Dict
from pydantic import BaseModel, Field, constr, confloat
from enum import Enum

class EducationLevel(str, Enum):
    OL = "OL"
    AL = "AL"
    UNDERGRADUATE = "UNDERGRADUATE"
    GRADUATE = "GRADUATE"
    MASTERS = "MASTERS"
    PHD = "PHD"

class Stream(str, Enum):
    PHYSICAL_SCIENCE = "PHYSICAL_SCIENCE"
    BIOLOGICAL_SCIENCE = "BIOLOGICAL_SCIENCE"
    COMMERCE = "COMMERCE"
    ARTS = "ARTS"

class DegreeType(str, Enum):
    BACHELORS = "Bachelors"
    MASTERS = "Masters"
    PHD = "PhD"

class RoleType(str, Enum):
    TECHNICAL = "Technical"
    RESEARCH = "Research"
    MANAGEMENT = "Management"

class ProjectType(str, Enum):
    RESEARCH = "Research"
    PROJECT = "Project"
    INTERNSHIP = "Internship"

class OLResults(BaseModel):
    mathematics: confloat(ge=0, le=100)
    science: confloat(ge=0, le=100)
    english: confloat(ge=0, le=100)
    first_language: Optional[confloat(ge=0, le=100)]
    ict: Optional[confloat(ge=0, le=100)]
    total_subjects_passed: int = Field(ge=0, le=9)
    core_subjects_average: confloat(ge=0, le=100)

class ALResults(BaseModel):
    stream: Stream
    subjects: Dict[str, confloat(ge=0, le=100)]
    zscore: Optional[float]

class Project(BaseModel):
    type: ProjectType
    domain: str
    duration_months: int = Field(ge=1)

class Internship(BaseModel):
    field: str
    duration_months: int = Field(ge=1)
    role_type: RoleType

class TechnicalCompetencies(BaseModel):
    programming: Optional[int] = Field(ge=1, le=5)
    data_analysis: Optional[int] = Field(ge=1, le=5)
    research: Optional[int] = Field(ge=1, le=5)
    domain_specific_tools: Dict[str, int] = Field(default_factory=dict)

class UniversityData(BaseModel):
    degree_type: DegreeType
    current_year: int = Field(ge=1, le=4)
    field_of_study: str
    specialization: Optional[str]
    current_gpa: Optional[confloat(ge=0, le=4.0)]
    major_specific_grades: Optional[Dict[str, float]]
    technical_competencies: Optional[TechnicalCompetencies]
    significant_projects: Optional[List[Project]]
    internships: Optional[List[Internship]]

class WorkPreferences(BaseModel):
    research_oriented: bool
    industry_oriented: bool
    entrepreneurship_interest: bool

class CareerGoals(BaseModel):
    further_studies: bool
    industry_experience: bool
    startup_plans: bool

class CareerPreferences(BaseModel):
    preferred_roles: List[str] = Field(min_items=3, max_items=3)
    preferred_sectors: List[str] = Field(min_items=3, max_items=3)
    work_preferences: WorkPreferences
    career_goals: CareerGoals

class StudentProfile(BaseModel):
    student_id: str
    education_level: EducationLevel
    ol_results: OLResults
    al_results: Optional[ALResults]
    university_data: Optional[UniversityData]
    skills_assessment: Dict[str, int] = Field(default_factory=dict)
    interests: List[str] = Field(min_items=3, max_items=3)
    career_preferences: Optional[CareerPreferences]
    constraints: Optional[Dict[str, any]]

class CareerPathScore(BaseModel):
    path_name: str
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reasons: List[str]
    requirements: List[str]
    next_steps: List[str]

class CareerGuidanceResponse(BaseModel):
    student_id: str
    timestamp: str
    top_career_paths: List[CareerPathScore]
    skill_gaps: Dict[str, List[str]]
    recommended_courses: List[str]
    growth_opportunities: List[str]

class UserBase(BaseModel):
    email: str
    full_name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    student_profile: Optional[StudentProfile]

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
