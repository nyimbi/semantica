# Test runner script (PowerShell)
# Runs tests with coverage reporting

$ErrorActionPreference = "Stop"

Write-Host "🧪 Running tests..." -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path "venv") {
    & .\venv\Scripts\Activate.ps1
}

# Check if tests directory exists
if (-not (Test-Path "tests")) {
    Write-Host "⚠️  No tests directory found" -ForegroundColor Yellow
    Write-Host "   Create tests/ directory and add test files" -ForegroundColor Yellow
    exit 0
}

# Check if test files exist
$testFiles = Get-ChildItem -Path tests -Recurse -Include "test_*.py", "*_test.py" -ErrorAction SilentlyContinue
if (-not $testFiles) {
    Write-Host "⚠️  No test files found" -ForegroundColor Yellow
    Write-Host "   Add test files to tests/ directory" -ForegroundColor Yellow
    exit 0
}

# Run tests with coverage
Write-Host "📊 Running pytest with coverage..." -ForegroundColor Yellow
pytest `
    --cov=semantica `
    --cov-report=html `
    --cov-report=term-missing `
    --cov-report=xml `
    -v `
    tests/

Write-Host ""
Write-Host "✅ Tests complete!" -ForegroundColor Green
Write-Host "📊 Coverage report: htmlcov/index.html" -ForegroundColor Cyan

