# TokenizerSimple

A lightweight, similarity-based custom text tokenizer written in Python.

## Overview

TokenizerSimple provides an algorithmic approach to parsing text into sub-word tokens without dependencies on heavy natural language processing (NLP) frameworks or machine learning models. It computes character-level prefix commonalities and mathematically splits words into logical base tokens and suffixes.

## Features

- **Prefix Matching**: Identifies common root prefixes between distinct words natively.
- **Suffix Extraction**: Extrapolates remaining word parts as standalone suffix tokens.
- **Dynamic Scoring Strategy**: Sorts and processes word pairs dynamically using a similarity heuristic based on prefix length relative to word length.
- **Standard Library Implementation**: Built strictly using Python's standard `re` module, requiring zero external dependencies.

## Architecture and Workflow

1. **Extraction**: Isolates unique lowercase words from the provided text string using regex boundary matching.
2. **Pair Evaluation**: Compares word pairs to identify the longest common prefix.
3. **Similarity Scoring**: Calculates a similarity metric and selectively filters the highest-scoring pairs.
4. **Token Generation**: If the score meets the specified similarity threshold (default `>= 0.5`), the prefix is established as a base token, and the remainder is separated as a suffix token. Unmatched words remain as entirely discrete units.
5. **Final Aggregation**: Combines all base tokens and suffix tokens into an optimized, unified array.

## Usage

Start the interactive script to test the tokenizer:

```bash
# 1. Activate the Python Virtual Environment
source venv/bin/activate
# Note: On Windows systems, execute `venv\Scripts\activate`

# 2. Execute the tokenizer script
python main.py
```

### Example Execution

```text
Enter your text: running runner sprint
Tokens: ['run', 'sprint', 'ning', 'ner']
```

## Project Structure

- `main.py` - Core algorithmic tokenizer and command-line execution script.
- `venv/` - Standard isolated Python virtual environment.