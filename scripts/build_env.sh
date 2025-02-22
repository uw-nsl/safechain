#!/bin/bash


# Check if a directory name is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <env_name>"
  exit 1
fi

ENV_NAME=$1
REQUIREMENTS_FILE="requirements.txt"

# Check if the requirements.txt file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements.txt does not exist"
  exit 1
fi

# Create the Conda environment
conda create --name $ENV_NAME python=3.10 -y

# Activate the Conda environment
source activate $ENV_NAME



# Install packages from requirements.txt using pip

pip install -r $REQUIREMENTS_FILE
pip install flash-attn --no-build-isolation
pip install git+https://github.com/dsbowen/strong_reject.git@main

echo "Environment $ENV_NAME set up with packages from $REQUIREMENTS_FILE"