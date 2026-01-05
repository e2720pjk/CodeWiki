#!/bin/bash

set -e

echo "Running Python linters..."

# Check import ordering
echo "Checking import order with isort..."
isort --check-only --diff codewiki/ test/ || {
    echo "Import ordering issues found. Run: isort codewiki/ test/"
    exit 1
}

# Check code formatting
echo "Checking code formatting with black..."
black --check --diff codewiki/ test/ || {
    echo "Code formatting issues found. Run: black codewiki/ test/"
    exit 1
}

# Run flake8 linter
echo "Running flake8..."
flake8 codewiki/ test/ --max-line-length=120 || {
    echo "Flake8 issues found"
    exit 1
}

echo "All lint checks passed!"
