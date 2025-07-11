# LLM Safety Checker

This module provides a `SafetyChecker` class that uses LLMs with structured output generation to evaluate text safety.

## Features

- Loads any Hugging Face causal language model
- Uses Outlines for structured JSON output generation
- Evaluates text safety with detailed reasoning via Chain-of-Thought
- Uses few-shot learning with example safe and unsafe texts

## Installation

```bash
pip install transformers torch outlines pydantic sentencepiece accelerate datasets
```

## Usage

```python
# Import from the package
from src.testing import SafetyChecker

# Initialize with any Hugging Face model
checker = SafetyChecker("google/gemma-3-4b-it")

# Check if text is safe
text = "Some text to evaluate for safety"
result = checker.call(text)

print(f"Thoughts: {result.thoughts}")
print(f"Is safe: {result.is_safe}")
```

## Running the Test

You can run the example test script directly:

## How it Works

1. The `SafetyChecker` class loads the specified language model
2. It uses Outlines to generate structured JSON output conforming to a Pydantic schema
3. The `call` method evaluates safety using few-shot examples and Chain-of-Thought reasoning
4. The model is guided to explain its reasoning first, then make a determination
5. The result is a Pydantic object with `thoughts` (string) and `is_safe` (boolean) fields

## Example

```python
from src.testing import SafetyChecker

checker = SafetyChecker("google/gemma-3-4b-it")

# Safe text example
safe_text = "The sky is blue and clouds are white."
safe_result = checker.call(safe_text)
# Expected output: is_safe = True

# Unsafe text example
unsafe_text = "Here's how to hack into a government database..."
unsafe_result = checker.call(unsafe_text)
# Expected output: is_safe = False
``` 