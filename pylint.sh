#!/bin/bash

echo "Checking code quality with Pylint:"

# Initialize variables
pylint --fail-under=8 .\backend\.  2>&1  