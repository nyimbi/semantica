# Code quality check script (PowerShell)
# Runs formatting, import sorting, linting, and type checking

$ErrorActionPreference = "Stop"

Write-Host "🔍 Running code quality checks..." -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path "venv") {
    & .\venv\Scripts\Activate.ps1
}

# Check if directories exist
if (-not (Test-Path "semantica")) {
    Write-Host "❌ semantica/ directory not found" -ForegroundColor Red
    exit 1
}

# Check formatting
Write-Host "📝 Checking code formatting (black)..." -ForegroundColor Yellow
try {
    black --check semantica/ 2>&1 | Out-Null
    Write-Host "✅ Black check passed" -ForegroundColor Green
} catch {
    Write-Host "❌ Code formatting issues found" -ForegroundColor Red
    Write-Host "   Run: black semantica/" -ForegroundColor Yellow
    exit 1
}

# Check import sorting
Write-Host "📦 Checking import sorting (isort)..." -ForegroundColor Yellow
try {
    isort --check-only semantica/ 2>&1 | Out-Null
    Write-Host "✅ isort check passed" -ForegroundColor Green
} catch {
    Write-Host "❌ Import sorting issues found" -ForegroundColor Red
    Write-Host "   Run: isort semantica/" -ForegroundColor Yellow
    exit 1
}

# Lint with flake8
Write-Host "🔎 Linting with flake8..." -ForegroundColor Yellow
try {
    flake8 semantica/
    Write-Host "✅ flake8 check passed" -ForegroundColor Green
} catch {
    Write-Host "❌ Linting issues found" -ForegroundColor Red
    exit 1
}

# Type check with mypy (non-blocking)
Write-Host "🔬 Type checking with mypy..." -ForegroundColor Yellow
try {
    mypy semantica/ 2>&1 | Out-Null
    Write-Host "✅ mypy check passed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Type checking issues found (non-blocking)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ All code quality checks passed!" -ForegroundColor Green

