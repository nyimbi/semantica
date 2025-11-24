# Development Scripts

Helper scripts for development, testing, and code quality checks.

## Available Scripts

### Setup Scripts

#### `setup-dev.sh` / `setup-dev.ps1`
Sets up the development environment.

**Bash (Linux/macOS):**
```bash
bash .github/scripts/setup-dev.sh
```

**PowerShell (Windows):**
```powershell
.github\scripts\setup-dev.ps1
```

**What it does:**
- Checks Python version
- Creates virtual environment if needed
- Installs package with dev dependencies
- Sets up pre-commit hooks
- Verifies installation

### Code Quality Scripts

#### `check-code.sh` / `check-code.ps1`
Runs all code quality checks (formatting, linting, type checking).

**Bash:**
```bash
bash .github/scripts/check-code.sh
```

**PowerShell:**
```powershell
.github\scripts\check-code.ps1
```

**What it checks:**
- Code formatting (black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy - non-blocking)

#### `format-code.sh` / `format-code.ps1`
Automatically formats code and sorts imports.

**Bash:**
```bash
bash .github/scripts/format-code.sh
```

**PowerShell:**
```powershell
.github\scripts\format-code.ps1
```

**What it does:**
- Formats code with black
- Sorts imports with isort

### Testing Scripts

#### `run-tests.sh` / `run-tests.ps1`
Runs tests with coverage reporting.

**Bash:**
```bash
bash .github/scripts/run-tests.sh
```

**PowerShell:**
```powershell
.github\scripts\run-tests.ps1
```

**What it does:**
- Runs pytest with coverage
- Generates HTML and terminal coverage reports
- Coverage report: `htmlcov/index.html`

## Quick Start

1. **Setup development environment:**
   ```bash
   # Linux/macOS
   bash .github/scripts/setup-dev.sh
   
   # Windows
   .github\scripts\setup-dev.ps1
   ```

2. **Before committing, run checks:**
   ```bash
   # Linux/macOS
   bash .github/scripts/check-code.sh
   
   # Windows
   .github\scripts\check-code.ps1
   ```

3. **If checks fail, format code:**
   ```bash
   # Linux/macOS
   bash .github/scripts/format-code.sh
   
   # Windows
   .github\scripts\format-code.ps1
   ```

4. **Run tests:**
   ```bash
   # Linux/macOS
   bash .github/scripts/run-tests.sh
   
   # Windows
   .github\scripts\run-tests.ps1
   ```

## Requirements

- Python 3.10+
- Virtual environment (created by setup script)
- Dev dependencies installed (`pip install -e ".[dev]"`)

## Notes

- All scripts automatically activate the virtual environment if it exists
- Scripts check for required directories before running
- Error messages provide helpful guidance
- Type checking (mypy) is non-blocking and won't fail the check

