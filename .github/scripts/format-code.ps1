# Code formatting script (PowerShell)
# Formats code with black and sorts imports with isort

$ErrorActionPreference = "Stop"

Write-Host "🎨 Formatting code..." -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path "venv") {
    & .\venv\Scripts\Activate.ps1
}

# Check if directories exist
if (-not (Test-Path "semantica")) {
    Write-Host "❌ semantica/ directory not found" -ForegroundColor Red
    exit 1
}

# Format with black
Write-Host "📝 Formatting with black..." -ForegroundColor Yellow
black semantica/
Write-Host "✅ Black formatting complete" -ForegroundColor Green

# Sort imports with isort
Write-Host "📦 Sorting imports with isort..." -ForegroundColor Yellow
isort semantica/
Write-Host "✅ Import sorting complete" -ForegroundColor Green

Write-Host ""
Write-Host "✅ Code formatting complete!" -ForegroundColor Green

