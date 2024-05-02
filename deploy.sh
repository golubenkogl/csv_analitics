#!/bin/bash

ENV_NAME=".venv"

# Check if virtual environment already exists
if [ ! -f "$ENV_NAME/bin/activate" ]; then
  echo "Creating a new virtual environment..."
  sudo apt-get install python3.10-venv
  python3 -m venv $ENV_NAME
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment if not already activated
if [[ $VIRTUAL_ENV == "" ]]; then
  echo "Activating the virtual environment..."
  source $ENV_NAME/bin/activate
else 
  echo "Virtual environment is already activated."
fi

# Upgrade pip to the latest version if needed
pip install --upgrade pip

# Install packages with constraints from "constraints.txt"
pip install -r $ENV_NAME/constraints.txt

echo "Deployment completed."
