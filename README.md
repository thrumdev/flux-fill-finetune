# Flux Fill Fine-Tuning

This repository provides a robust, experiment-ready training script for fine-tuning the [Flux Fill (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) model using Hugging Face Diffusers, Accelerate, and Weights & Biases (wandb). The script supports modular training, validation, experiment tracking, checkpointing, and reproducibility.

## Features
- Modular training and validation loops
- Experiment tracking and image logging with wandb
- Distributed and mixed-precision training with Accelerate
- Full checkpointing and resuming
- Configurable random seed for reproducibility
- CLI for all major training and experiment parameters

## Directory Structure
```
flux-fill-finetuning/
├── main.py              # Main training script
├── requirements.txt     # Python dependencies
├── checkpoints/         # Directory for saving model checkpoints
├── data/
│   ├── training/
│   │   ├── images/
│   │   ├── masks/
│   │   └── prompts/
│   └── validation/
│       ├── images/
│       ├── masks/
│       └── prompts/
```

## Initial Setup

### 1. Install Python Dependencies
```fish
pip install -r requirements.txt
```

### 2. Set Up Weights & Biases (wandb)
- Create a free account at [wandb.ai](https://wandb.ai/).
- Log in from the command line (only needed once per machine):
```fish
wandb login
```
- Paste your API key when prompted.

### 3. Set Up Accelerate (for distributed/mixed-precision training)
- Run the Accelerate config wizard (only needed once per machine):
```fish
accelerate config
```
- Answer the prompts to match your hardware and preferences.

### 4. Prepare Your Data
- Organize your data as follows:
  - `data/training/images/`, `data/training/masks/`, `data/training/prompts/`
  - `data/validation/images/`, `data/validation/masks/`, `data/validation/prompts/`
- Each image/mask should have a matching filename (e.g., `0001.png` and `0001.txt` for the prompt).

## Usage

Run the training script with your desired parameters:
```fish
python main.py \
  --wandb_project flux-fill-finetuning \
  --wandb_name my-experiment-1 \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --validation_epochs 1 \
  --save-epochs 2 \
  --seed 42
```

**Arguments:**
- `--wandb_project` (required): Name of the wandb project (umbrella for all runs)
- `--wandb_name` (required): Name for this specific experiment/run
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--validation_epochs`: How often (in epochs) to run validation
- `--save-epochs`: How many epochs between checkpoints (0 = only final)
- `--seed`: Random seed (optional; if not set, a random one is generated and printed)

## Checkpoints & Resuming
- Checkpoints are saved in the `checkpoints/` directory at the specified interval and always at the end.
- To resume from a checkpoint, you can extend the script to load from a saved `.pt` file using the provided `load_checkpoint` function.

## Experiment Tracking
- All training/validation metrics and sample images are logged to wandb.
- You can view and compare runs at [wandb.ai](https://wandb.ai/).

## Notes
- The script uses `drop_last=False` for DataLoaders by default. If your model requires fixed batch sizes, you can set `drop_last=True` in the script.
- For best reproducibility, always set a seed with `--seed`.

## License
This project is intended for research and educational purposes.
