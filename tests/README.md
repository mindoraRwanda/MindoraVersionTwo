# Tests Module

This module contains all tests for the Mindora application, organized in a structured and maintainable way.

## Structure

```
tests/
├── __init__.py              # Test utilities and configuration
├── conftest.py              # Pytest fixtures and configuration
├── pytest.ini              # Pytest configuration
├── requirements.txt         # Testing dependencies
├── README.md               # This file
├── unit/                   # Unit tests
│   ├── __init__.py
│   ├── test_query_validator.py
│   ├── test_prompts.py
│   └── test_langgraph_validator.py
└── integration/            # Integration tests
    ├── __init__.py
    └── test_langgraph_workflow.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
Unit tests focus on individual components and functions in isolation.

- **test_query_validator.py**: Tests for both keyword-based and LangGraph-based query validators
- **test_prompts.py**: Tests for all prompt modules and their functionality
- **test_langgraph_validator.py**: Unit tests for LangGraph query validator components

### Integration Tests (`tests/integration/`)
Integration tests verify that multiple components work together correctly.

- **test_langgraph_workflow.py**: Tests for the complete LangGraph workflow

## Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install -r tests/requirements.txt
```

### Run All Tests
```bash
# From the project root directory
pytest tests/

# Or from the tests directory
cd tests && pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_query_validator.py

# Specific test class
pytest tests/unit/test_query_validator.py::TestQueryValidatorService

# Specific test method
pytest tests/unit/test_query_validator.py::TestQueryValidatorService::test_validate_mental_support_query
```

### Run Tests with Markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only tests that require LLM
pytest -m llm

# Run only tests that require database
pytest -m database

# Skip slow tests
pytest -m "not slow"
```

### Run Tests with Coverage
```bash
# Generate coverage report
pytest --cov=app --cov-report=html tests/

# Generate coverage report for specific modules
pytest --cov=app.services --cov-report=term-missing tests/unit/test_query_validator.py
```

### Run Tests in Parallel
```bash
# Run tests in parallel
pytest -n auto tests/

# Run tests with specific number of workers
pytest -n 4 tests/
```

## Test Configuration

### Markers
- `slow`: Tests that take a long time to run
- `integration`: Integration tests
- `unit`: Unit tests
- `llm`: Tests that require LLM services
- `database`: Tests that require database
- `mock`: Tests that use mocking
- `async`: Async tests
- `performance`: Performance tests

### Fixtures
Common fixtures are defined in `conftest.py`:

- `test_config`: Global test configuration
- `mock_llm_provider`: Mock LLM provider for testing
- `mock_llm_service`: Mock LLM service for testing
- `test_user`: Test user data
- `test_conversation`: Test conversation data
- `test_query`: Test query data
- `langgraph_validator`: LangGraph validator instance
- `mock_database`: Mock database session

### Test Data
Test utilities in `tests/__init__.py` provide helper functions:

- `TestUtils.create_mock_user()`: Create mock user data
- `TestUtils.create_mock_conversation()`: Create mock conversation data
- `TestUtils.create_mock_query()`: Create mock query data
- `TestFixtures.mock_llm_response()`: Create mock LLM responses
- `TestFixtures.mock_crisis_detection()`: Create mock crisis detection results

## Writing Tests

### Test File Naming
- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

### Test Structure
```python
import pytest
from app.module import FunctionUnderTest

class TestFunctionUnderTest:
    """Test class for FunctionUnderTest."""

    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        return {"key": "value"}

    def test_success_case(self, setup_data):
        """Test successful execution."""
        result = FunctionUnderTest(setup_data)
        assert result.success == True

    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            FunctionUnderTest(None)

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test async function."""
        result = await async_function_under_test()
        assert result is not None
```

### Best Practices
1. **Test one thing at a time**: Each test should verify one specific behavior
2. **Use descriptive names**: Test names should clearly describe what they test
3. **Use fixtures**: Extract common setup code into fixtures
4. **Mock external dependencies**: Use mocks for external services and databases
5. **Test edge cases**: Include tests for edge cases and error conditions
6. **Use assertions effectively**: Use appropriate assertions for different types of checks
7. **Document complex tests**: Add comments explaining complex test logic

## Continuous Integration

### GitHub Actions
Tests are automatically run on:
- Push to main branch
- Pull requests
- Manual workflow dispatch

### Test Reports
- Coverage reports are generated in HTML format
- Test results are available in the Actions tab
- Failed tests are reported with detailed error messages

## Debugging Tests

### Debug Failing Tests
```bash
# Run specific test with verbose output
pytest tests/unit/test_query_validator.py::TestQueryValidatorService::test_validate_mental_support_query -v -s

# Run test with debugger
pytest tests/unit/test_query_validator.py::TestQueryValidatorService::test_validate_mental_support_query --pdb

# Run test with maximum verbosity
pytest tests/unit/test_query_validator.py -vv
```

### Common Issues
1. **Import errors**: Ensure all dependencies are installed
2. **Async test issues**: Use `@pytest.mark.asyncio` for async tests
3. **Fixture scope issues**: Check fixture scope in `conftest.py`
4. **Mock issues**: Ensure mocks are properly configured

## Contributing

When adding new tests:
1. Follow the existing structure and naming conventions
2. Add appropriate markers for test categorization
3. Include docstrings for test classes and methods
4. Use fixtures for common setup code
5. Test both success and failure scenarios
6. Update this README if adding new test categories

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)