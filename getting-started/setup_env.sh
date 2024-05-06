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
pip install transformers datasets accelerate peft bitsandbytes ipywidgets evaluate huggingface_hub nvidia-ml-py3 
pip install matplotlib scikit-learn pytest pre-commit torchtune
pip install openai tiktoken trl ipykernel anthropic PyPDF2 seaborn
pip install git+https://github.com/pharringtonp19/trics.git

echo "Setup complete and the virtual environment $ENV_NAME is ready to use."
