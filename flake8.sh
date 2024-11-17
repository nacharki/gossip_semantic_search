#!/bin/bash

echo "Checking code quality with flake8:"

# Run flake8 on all Python files in backend/ and frontend/ directories
flake8 backend/ frontend/ --ignore=E501,W503
