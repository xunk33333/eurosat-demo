# EuroSAT Classification Demo

This project demonstrates image classification on the EuroSAT dataset using both Vision Transformer (ViT) and ResNet models.

## Features
- Supports both ViT and ResNet models (configurable)
- Training and validation pipeline  
- TensorBoard logging
- Model checkpointing

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

3. View TensorBoard logs:
```bash
tensorboard --logdir runs
```

## Configuration
Modify `config.py` to:
- Switch between ViT and ResNet (model_type)
- Adjust model hyperparameters
- Change training settings

## File Structure
- `data_loader.py`: Data loading and preprocessing
- `model.py`: Model definitions (ViT and ResNet)
- `train.py`: Training pipeline
- `config.py`: Configuration settings
