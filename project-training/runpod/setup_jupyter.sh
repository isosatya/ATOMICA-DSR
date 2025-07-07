#!/bin/bash

# Setup script for running Jupyter Lab in RunPod
# This allows interactive access to notebooks through browser

echo "Setting up Jupyter Lab for interactive notebook access..."

# Install Jupyter Lab if not already installed
pip install jupyterlab ipywidgets

# Create Jupyter config directory
mkdir -p ~/.jupyter

# Generate Jupyter config
jupyter lab --generate-config

# Set password for Jupyter Lab (optional - you can set this interactively)
# echo "from jupyter_server.auth import passwd; print(passwd())" | python3

# Create a script to start Jupyter Lab
cat > ~/start_jupyter.sh << 'EOF'
#!/bin/bash

# Get the pod's public IP (RunPod provides this as an environment variable)
PUBLIC_IP=${RUNPOD_PUBLIC_IP:-$(curl -s ifconfig.me)}

echo "Starting Jupyter Lab..."
echo "Public IP: $PUBLIC_IP"
echo "You can access Jupyter Lab at: http://$PUBLIC_IP:8888"
echo "Use the token that will be displayed below to log in"
echo ""

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
EOF

chmod +x ~/start_jupyter.sh

echo "Setup complete!"
echo ""
echo "To start Jupyter Lab, run:"
echo "  ./start_jupyter.sh"
echo ""
echo "Or run directly:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "Then access it in your browser at: http://YOUR_POD_IP:8888"
echo ""
echo "Note: You may need to configure port forwarding in RunPod to access port 8888" 