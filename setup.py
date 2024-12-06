from setuptools import setup, find_packages

setup(
    name="next_step",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.1",
        "joblib>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.6b0",
            "isort>=5.9.2",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ],
    },
    author="Adithya Ekanayaka",
    author_email="adithyasean@gmail.com",
    description="AI-powered career guidance system for Sri Lankan students",
    keywords="career-guidance, machine-learning, education",
    python_requires=">=3.8",
)
