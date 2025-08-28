# Contributing to the Project

## 🤝 How to Contribute

Thank you for your interest in contributing to the Book Recommender AI Agent! Your help is very valuable.

## 🚀 Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/your-username/Book-recommender-AI-agent.git
cd Book-recommender-AI-agent
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Setup pre-commit hooks (optional)
pre-commit install
```

### 3. Project Structure
- `app/graph/`: LangGraph agent logic
- `app/api/`: FastAPI API
- `ui/`: Streamlit interface
- `tests/`: Tests and evaluations
- `notebooks/`: Jupyter notebooks for development

## 🧪 Running Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Specific evaluations
python -m tests.nodes_evals.router
python -m tests.nodes_evals.recommend_books
```

## 📝 Contribution Guidelines

### Types of Contributions
- 🐛 **Bug fixes**: Error corrections
- ✨ **Features**: New functionalities
- 📚 **Documentation**: Documentation improvements
- 🧪 **Tests**: New tests or evaluations
- 🎨 **UI/UX**: Interface improvements

### Contribution Process
1. **Create Issue**: Describe the problem or feature
2. **Create Branch**: `git checkout -b feature/my-feature`
3. **Develop**: Implement the changes
4. **Tests**: Make sure tests pass
5. **Commit**: Use descriptive messages
6. **Pull Request**: Describe your changes

### Code Conventions
- **Python**: Follow PEP 8
- **Commits**: Use conventional commits
- **Docstrings**: Document functions and classes
- **Type hints**: Use type annotations

### Commit Examples
```
feat: add publication year filter
fix: correct duplicate recommendations error
docs: update README with new instructions
test: add tests for router node
```

## 🎯 Areas That Need Help

### High Priority
- [ ] Improve recommendation algorithm
- [ ] Add more book data sources
- [ ] Optimize API performance
- [ ] Improve frontend UI/UX

### Medium Priority
- [ ] Add multi-language support
- [ ] Implement rating system
- [ ] Add advanced filters
- [ ] Improve evaluation system

### Low Priority
- [ ] Add dark theme
- [ ] Implement notifications
- [ ] Add data export
- [ ] Improve documentation

## 🐛 Reporting Bugs

When reporting a bug, include:
- **Description**: What you expected vs what happened
- **Steps**: How to reproduce the error
- **Environment**: OS, Python version, etc.
- **Logs**: Relevant error messages

## 💡 Requesting Features

For new features:
- **Context**: Why it's needed
- **Description**: How it should work
- **Use cases**: Specific examples
- **Impact**: Who it would help

## 📞 Contact

- GitHub Issues: For bugs and features
- Discussions: For general questions
- Email: [contact] for sensitive matters

We look forward to your contributions! 🚀
