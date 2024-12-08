from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .database import Base, engine

app = FastAPI(
    title="Next Step AI Career Guidance",
    description="API for career guidance recommendations using machine learning",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    """Initialize application."""
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Next Step AI Career Guidance API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
