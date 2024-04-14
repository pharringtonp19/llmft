#!/bin/bash

# Change directory
cd llmft

# Name of the virtual environment
ENV_NAME="llms"

# Create virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Install any necessary Python packages
pip3 install torch torchvision torchaudio
pip install transformers datasets accelerate peft bitsandbytes ipywidgets evaluate huggingface_hub 
pip install matplotlib markdown

echo "Setup complete and the virtual environment $ENV_NAME is ready to use."
