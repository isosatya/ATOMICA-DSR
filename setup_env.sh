#!/bin/bash

# Install X11 libraries needed for RDKit on headless servers
apt-get update && apt-get install -y libxrender1 libxext6 libx11-6 libxrandr2 libxinerama1 libxi6 libxss1 libxtst6

ENVNAME=atomicadsr
conda create -n $ENVNAME python=3.9 -y
source activate $ENVNAME
conda install "numpy<2" -y
pip3 install torch==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install torch_scatter torch_cluster --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html
pip3 install tensorboard==2.18.0
pip3 install e3nn==0.5.1 # possibly not compatible with e3nn > 0.5.4
pip3 install scipy==1.13.1
pip3 install rdkit-pypi==2022.9.5
pip3 install openbabel-wheel==3.1.1.20
pip3 install biopython==1.84
pip3 install biotite==0.40.0
pip3 install atom3d
pip3 install wandb==0.18.2
pip3 install orjson
pip3 install huggingface_hub

# plotting
pip3 install umap-learn
pip3 install matplotlib
pip3 install seaborn
pip3 install plotly