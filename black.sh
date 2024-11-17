#!/bin/bash
echo "Checking code quality with black:"

# Run black on all Python files in backend/ and frontend/ directories
black backend/ frontend/ --line-length=120
