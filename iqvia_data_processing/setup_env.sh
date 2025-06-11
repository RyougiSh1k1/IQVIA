#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "$HOME/iqvia_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $HOME/iqvia_env
    
    # Activate the environment
    source $HOME/iqvia_env/bin/activate
    
    # Install required packages
    pip install --upgrade pip
    pip install pandas numpy scikit-learn matplotlib seaborn tqdm
else
    echo "Virtual environment already exists"
fi

echo "Environment setup complete!"
echo "To activate, run: source $HOME/iqvia_env/bin/activate"