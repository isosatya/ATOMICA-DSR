![ATOMICA logo](assets/atomica_logo.png)
# Learning Universal Representations of Intermolecular Interactions

**Authors**
* Ada Fang
* Zaixi Zhang
* Andrew Zhou
* Marinka Zitnik

[Preprint](https://www.biorxiv.org/content/10.1101/2025.04.02.646906v1) | [Project Website](https://zitniklab.hms.harvard.edu/projects/ATOMICA)

ATOMICA is a geometric AI model that learns universal representations of molecular interactions at an atomic scale. The model is pretrained on 2,037,972 molecular interaction interfaces from the Protein Data Bank and Cambridge Structural Database, this includes protein-small molecule, protein-ion, small molecule-small molecule, protein-protein, protein-peptide, protein-RNA, protein-DNA, and nucleic acid-small molecule complexes. Embeddings of ATOMICA can be generated with the open source model weights and code to be used for various downstream tasks. In the paper, we demonstrate the utility of ATOMICA embeddings for studying the human interfaceome network with ATOMICANets and for annotating ions and small molecules to proteins in the dark proteome.

## :rocket: Installation and Setup

### 1. Download the Repository
Clone the Gihub Repository:
```bash
git clone https://github.com/mims-harvard/ATOMICA
cd ATOMICA
```

### 2. Set Up Environment

**Option A: Local Setup**
Set up the environment according to `setup_env.sh`.

**Option B: RunPod Cloud Deployment**
For cloud deployment on RunPod, see [project-training/runpod/RUNPOD_README.md](project-training/runpod/RUNPOD_README.md) for detailed instructions. This includes:
- Docker container with all dependencies pre-installed
- CUDA support for GPU acceleration
- Jupyter notebook access
- Automated environment setup

### 3. (optional) Download Processed Datasets
The data for pretraining and downstream analyses is hosted at [Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX).

We provide the following datasets:
* Processed CSD and QBioLiP (based on PDB) interaction complex graphs for pretraining
* Processed protein interfaces of human proteome binding sites to ion, small molecule, lipid, nucleic acid, and protein modalities
* Processed protein interfaces of dark proteome binding sites to ion and small molecules

### 4. Download Model Checkpoints
Model checkpoints are provided on [Hugging Face](https://huggingface.co/ada-f/ATOMICA). The following models are available:
* ATOMICA model
* Pretrained ATOMICA-Interface model
* Finetuned ATOMICA-Ligand prediction models for the following ligands:
    * metal ions: Ca, Co, Cu, Fe, K, Mg, Mn, Na, Zn
    * small molecules: ADP, ATP, GTP, GDP, FAD, NAD, NAP, NDP, HEM, HEC, CIT, CLA

## :star: Usage
### Train ATOMICA
Training scripts for pretraining ATOMICA and finetuning ATOMICA-Interface and ATOMICA-Ligand are provided in `scripts/`.

### Inference with ATOMICA-Ligand
Refer to the jupyter notebook at `case_studies/atomica_ligand/example_run_atomica_ligand.ipynb` for an example of how to use the model for binder prediction.

### Explore ATOMICANets
Refer to the jupyter notebook at `case_studies/atomica_net/example_atomica_net.ipynb`

### Embedding your own structures
Make sure to download the ATOMICA model weights and config files from [Hugging Face](https://huggingface.co/ada-f/ATOMICA).

**For embedding biomolecular complexes:** process .pdb files with `data/process_pdbs.py` and embed with `get_embeddings.py`. See further details for data processing in the `data/README.md` file [here](https://github.com/mims-harvard/ATOMICA/tree/main/data).

**For embedding protein-(ion/small molecule/lipid/nucleic acid/protein) interfaces:** first predict (ion/small molecule/lipid/nucleic acid/protein) binding sites with [PeSTo](https://github.com/LBM-EPFL/PeSTo), second process the PeSTo output .pdb files with `data/process_PeSTo_results.py`, finally embed with `get_embeddings.py`.

## :bulb: Questions
For questions, please leave a GitHub issue or contact Ada Fang at <ada_fang@g.harvard.edu>.

## :balance_scale: License
The code in this package is licensed under the MIT License.

## :scroll: Citation
If you use ATOMICA in your research, please cite the following [preprint](https://www.biorxiv.org/content/10.1101/2025.04.02.646906v1):
```
@article{Fang2025ATOMICA,
  author = {Fang, Ada and Zhang, Zaixi and Zhou, Andrew and Zitnik, Marinka},
  title = {ATOMICA: Learning Universal Representations of Intermolecular Interactions},
  year = {2025},
  journal = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  doi = {10.1101/2025.04.02.646906},
  url = {https://www.biorxiv.org/content/10.1101/2025.04.02.646906v1},
  note = {preprint},
}
```

# ATOMICA-DSR

ATOMICA (ATOMIC Interaction and Conformational Analysis) is a deep learning framework for molecular interaction modeling and drug discovery.

## 🚀 Quick Start

### Local Setup

1. **Clone the repository** (requires authentication for private repos):
   ```bash
   # Option A: Using Personal Access Token (Recommended)
   git clone https://YOUR_TOKEN@github.com/your-username/ATOMICA-DSR.git
   
   # Option B: Using SSH
   git clone git@github.com:your-username/ATOMICA-DSR.git
   ```

2. **Set up environment**:
   ```bash
   cd ATOMICA-DSR
   bash setup_env.sh
   ```

3. **Download pre-trained models**:
   ```bash
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ada-f/ATOMICA', filename='ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt', local_dir='project-training/original-model-config')"
   ```

### 🐳 RunPod Deployment (Recommended for GPU Training)

For GPU-accelerated training, we recommend using RunPod with our pre-built Docker image:

#### Option 1: Use Pre-built Docker Image
1. **Deploy on RunPod**:
   - Create new pod with custom image: `andress777/atomica-runpod:latest`
   - Choose GPU: RTX 4090, RTX 3090, or A100
   - Set port: 8888 (for Jupyter)

2. **Start training**:
   ```bash
   # Download model weights
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ada-f/ATOMICA', filename='ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt', local_dir='project-training/original-model-config')"
   
   # Run training
   python train.py --task PDBBind --train_set project-training/data/train_items.pkl --valid_set project-training/data/val_items.pkl --gpus 0 --max_epoch 1000 --save_dir project-training/model_checkpoints/pdbind_training
   ```

#### Option 2: Build Your Own Docker Image
```bash
# Build locally
docker build -t atomica-runpod -f project-training/runpod/Dockerfile .

# Push to Docker Hub
docker tag atomica-runpod your-username/atomica-runpod:latest
docker push your-username/atomica-runpod:latest

# Use on RunPod with custom image: your-username/atomica-runpod:latest
```

**For detailed RunPod instructions, see**: [`project-training/runpod/RUNPOD_README.md`](project-training/runpod/RUNPOD_README.md)

## 🔐 GitHub Authentication

This repository requires authentication for cloning. Choose one of these methods:

### Personal Access Token (Recommended)
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with 'repo' scope
3. Use: `git clone https://YOUR_TOKEN@github.com/your-username/ATOMICA-DSR.git`

### SSH Keys
1. Ensure your SSH key is added to GitHub
2. Use: `git clone git@github.com:your-username/ATOMICA-DSR.git`

### Download as ZIP
1. Download from GitHub web interface
2. Extract and rename to `ATOMICA-DSR`

## 📚 Documentation

- **Training Guide**: [`project-training/TRAINING_GUIDE.md`](project-training/TRAINING_GUIDE.md)
- **RunPod Deployment**: [`project-training/runpod/RUNPOD_README.md`](project-training/runpod/RUNPOD_README.md)
- **Case Studies**: [`case_studies/`](case_studies/)

## 🏗️ Project Structure

```
ATOMICA-DSR/
├── data/                    # Data processing scripts
├── models/                  # Model architectures
├── trainers/                # Training utilities
├── utils/                   # Utility functions
├── case_studies/           # Example notebooks
├── project-training/       # Training configurations
│   ├── runpod/            # RunPod deployment files
│   │   ├── Dockerfile     # Docker configuration
│   │   ├── setup_runpod.sh # Manual setup script
│   │   └── RUNPOD_README.md # RunPod guide
│   └── TRAINING_GUIDE.md  # Training instructions
└── README.md              # This file
```

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0.1+ with CUDA support
- e3nn 0.5.1
- RDKit
- Biotite
- Atom3D

## 📖 Usage Examples

### Training
```bash
python train.py \
  --task PDBBind \
  --train_set data/train.pkl \
  --valid_set data/valid.pkl \
  --gpus 0 \
  --max_epoch 1000 \
  --save_dir model_checkpoints
```

### Inference
```bash
python get_embeddings.py \
  --model_config model_config.json \
  --model_weights model_weights.pt \
  --data_path data.pkl \
  --output_path embeddings.pkl
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original ATOMICA paper and implementation
- PyTorch Geometric team
- e3nn contributors