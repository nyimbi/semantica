# Development environment setup script (PowerShell)
# Sets up Python virtual environment and installs dependencies

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up development environment..." -ForegroundColor Cyan

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    Write-Host "📦 $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10 or higher" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install project in editable mode with dev dependencies
Write-Host "📥 Installing package with dev dependencies..." -ForegroundColor Yellow
pip install -e ".[dev]"

# Install pre-commit hooks if available
try {
    $precommit = Get-Command pre-commit -ErrorAction SilentlyContinue
    if ($precommit) {
        Write-Host "🪝 Installing pre-commit hooks..." -ForegroundColor Yellow
        pre-commit install
    } else {
        Write-Host "⚠️  pre-commit not found, skipping hooks installation" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Pre-commit installation skipped" -ForegroundColor Yellow
}

# Verify installation
Write-Host "✅ Verifying installation..." -ForegroundColor Yellow
try {
    python -c "import semantica; print(f'Semantica version: {semantica.__version__}')"
    Write-Host "✅ Installation verified" -ForegroundColor Green
} catch {
    Write-Host "❌ Installation verification failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✨ Development environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1          # Activate virtual environment" -ForegroundColor White
Write-Host "  pytest                              # Run tests" -ForegroundColor White
Write-Host "  black semantica/                    # Format code" -ForegroundColor White
Write-Host "  .github\scripts\check-code.ps1       # Run all checks" -ForegroundColor White

