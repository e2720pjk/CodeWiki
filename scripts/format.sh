#!/bin/bash

set -e

echo "Auto-formatting Python code..."

# Format imports
echo "Organizing imports..."
isort codewiki/ test/

# Format code
echo "Formatting code..."
black codewiki/ test/

echo "Code formatted successfully!"
echo "Run ./scripts/lint.sh to verify formatting"
