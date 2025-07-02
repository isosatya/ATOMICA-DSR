# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
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
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY project-training/runpod/requirements.txt .
RUN pip install -r requirements.txt

# Install PyTorch Geometric dependencies
RUN pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/project-training/model_checkpoints
RUN mkdir -p /workspace/project-training/datasets
RUN mkdir -p /workspace/project-training/outputs
RUN mkdir -p /workspace/project-training/logs
RUN mkdir -p /workspace/project-training/data
RUN mkdir -p /workspace/project-training/original-model-config

# Set default command
CMD ["/bin/bash"] 