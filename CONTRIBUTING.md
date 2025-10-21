# 🤝 Contributing to Semantica

Thank you for your interest in contributing to Semantica! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### **Types of Contributions**

1. **🐛 Bug Reports**: Report bugs and issues
2. **✨ Feature Requests**: Suggest new features and improvements
3. **📝 Documentation**: Improve documentation and examples
4. **💻 Code Contributions**: Submit code changes and improvements
5. **🧪 Testing**: Add tests and improve test coverage
6. **🌐 Community**: Help with community support and discussions

## 🚀 Getting Started

### **Prerequisites**
- Python 3.8 or higher
- Git
- Basic knowledge of Python and semantic web technologies

### **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/semantica.git
cd semantica

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

## 📝 Development Workflow

### **1. Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### **2. Make Your Changes**
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### **3. Test Your Changes**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantica

# Run linting
flake8 semantica/
mypy semantica/

# Format code
black semantica/
isort semantica/
```

### **4. Commit Your Changes**
```bash
git add .
git commit -m "feat: add new PDF processor functionality

- Add support for table extraction from PDFs
- Implement metadata extraction
- Add comprehensive tests
- Update documentation"
```

### **5. Push and Create Pull Request**
```bash
git push origin feature/your-feature-name
```

## 📋 Coding Standards

### **Python Code Style**
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters (Black default)

### **Code Quality**
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use [mypy](https://mypy.readthedocs.io/) for type checking
- Maintain test coverage above 80%
- Write docstrings for all public functions and classes

### **Commit Message Format**
Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(processors): add Excel file processor
fix(core): resolve memory leak in knowledge graph builder
docs(api): update API documentation for new features
test(extraction): add tests for entity extraction
```

## 🧪 Testing Guidelines

### **Test Structure**
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Performance tests in `tests/performance/`
- Test data in `tests/fixtures/`

### **Writing Tests**
```python
import pytest
from semantica.processors.document.pdf_processor import PDFProcessor

class TestPDFProcessor:
    def setup_method(self):
        self.processor = PDFProcessor({
            'extract_tables': True,
            'extract_images': False
        })
    
    def test_can_process_pdf(self):
        """Test that PDF processor can identify PDF files."""
        assert self.processor.can_process("document.pdf")
        assert not self.processor.can_process("document.txt")
    
    def test_process_pdf(self, sample_pdf_path):
        """Test PDF processing functionality."""
        result = self.processor.process(sample_pdf_path)
        assert result.content is not None
        assert len(result.metadata) > 0
```

### **Test Requirements**
- All new code must have corresponding tests
- Maintain test coverage above 80%
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies

## 📚 Documentation Guidelines

### **Code Documentation**
- Use Google-style docstrings
- Include type hints for all functions
- Document all public APIs

```python
def extract_entities(self, text: str) -> List[Entity]:
    """Extract named entities from text.
    
    Args:
        text: Input text to extract entities from.
        
    Returns:
        List of extracted entities with confidence scores.
        
    Raises:
        ValueError: If text is empty or None.
    """
    pass
```

### **Documentation Updates**
- Update README.md for new features
- Add examples in `examples/` directory
- Update API documentation
- Create tutorials for complex features

## 🔍 Review Process

### **Pull Request Checklist**
- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] Commit messages follow conventional format
- [ ] No breaking changes (or clearly documented)

### **Review Guidelines**
- Be respectful and constructive
- Focus on code quality and functionality
- Suggest improvements when possible
- Test the changes locally if needed

## 🐛 Bug Reports

### **Bug Report Template**
```markdown
**Bug Description**
Brief description of the bug.

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 10, macOS 12.0]
- Python version: [e.g., 3.9.7]
- Semantica version: [e.g., 0.1.0]

**Additional Information**
Any other relevant information.
```

## 💡 Feature Requests

### **Feature Request Template**
```markdown
**Feature Description**
Brief description of the feature.

**Use Case**
Why this feature would be useful.

**Proposed Implementation**
How you think it could be implemented.

**Alternatives Considered**
Other approaches you've considered.
```

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

## 🎉 Recognition

Contributors will be recognized in:
- Repository contributors list
- Release notes
- Documentation acknowledgments
- Community highlights

## 📞 Getting Help

- **Discussions**: [GitHub Discussions](https://github.com/semantica/semantica/discussions)
- **Issues**: [GitHub Issues](https://github.com/semantica/semantica/issues)
- **Discord**: [Community Discord](https://discord.gg/semantica)
- **Email**: team@semantica.io

## 📄 License

By contributing to Semantica, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Semantica! 🚀 