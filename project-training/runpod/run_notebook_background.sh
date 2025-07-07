#!/bin/bash

# Script to run Jupyter notebook in background using screen
# This allows you to detach and reattach to the session

echo "Setting up Jupyter Lab in background session..."

# Install Jupyter Lab if not already installed
pip install jupyterlab ipywidgets

# Create a screen session for Jupyter
screen -dmS jupyter_session bash -c "
    echo 'Starting Jupyter Lab in screen session...'
    echo 'To attach to this session, run: screen -r jupyter_session'
    echo 'To detach from session, press: Ctrl+A, then D'
    echo ''
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
"

echo "Jupyter Lab started in background session 'jupyter_session'"
echo ""
echo "To attach to the session and see the output:"
echo "  screen -r jupyter_session"
echo ""
echo "To list all screen sessions:"
echo "  screen -ls"
echo ""
echo "To kill the session:"
echo "  screen -S jupyter_session -X quit"
echo ""
echo "Access Jupyter Lab at: http://YOUR_POD_IP:8888" 