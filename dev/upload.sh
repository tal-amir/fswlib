#!/bin/bash

# Determine repository
REPO="testpypi"
REPO_URL="https://test.pypi.org/legacy/"

if [[ "$1" == "--real" ]]; then
  REPO="pypi"
  REPO_URL="https://upload.pypi.org/legacy/"
fi

# Print target and ask for confirmation
echo "Code will be uploaded to: $REPO_URL"
read -p "Type 'yes' to proceed: " CONFIRM

if [[ "$CONFIRM" == "yes" ]]; then
  python3 -m twine upload --repository $REPO dist/*
else
  echo "Upload cancelled."
fi
