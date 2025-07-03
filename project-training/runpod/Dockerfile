# Use Ubuntu base image and install CUDA manually
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    libgsl-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libfftw3-dev \
    liblapack-dev \
    libblas-dev \
    libgfortran5 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libgl1-mesa-glx \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA support FIRST
RUN pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric dependencies
RUN pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Copy requirements and install remaining Python dependencies
COPY project-training/runpod/requirements.txt .
RUN pip install -r requirements.txt

# Install additional dependencies
RUN pip install huggingface_hub

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/project-training/model_checkpoints
RUN mkdir -p /workspace/project-training/datasets
RUN mkdir -p /workspace/project-training/outputs
RUN mkdir -p /workspace/project-training/logs
RUN mkdir -p /workspace/project-training/data
RUN mkdir -p /workspace/project-training/original-model-config

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/workspace"

# Set default command
CMD ["/bin/bash"] 