# PowerShell setup script for Next Step AI

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Next Step AI development environment..."

# Check Python version
$python_version = (python -V 2>&1).ToString().Split(" ")[1]
Write-Host "📌 Using Python version: $python_version"

# Remove existing venv if it exists
if (Test-Path "venv") {
    Write-Host "🧹 Removing existing virtual environment..."
    Remove-Item -Recurse -Force venv
}

# Create virtual environment
Write-Host "� Creating virtual environment..."
python -m venv venv

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Upgrade pip and install build dependencies first
Write-Host "🔄 Upgrading pip and installing build dependencies..."
python -m pip install --upgrade setuptools wheel

# Install all requirements
Write-Host "� Installing project dependencies..."
python -m pip install --upgrade -r requirements.txt

# Install package in development mode
Write-Host "🛠️ Installing package in development mode..."
python -m pip install -e .

# Set up pre-commit hooks if git is initialized
if (Test-Path ".git") {
    Write-Host "🔩 Setting up pre-commit hooks..."
    pre-commit install
}

Write-Host "✨ Setup complete! The virtual environment is now activated."
