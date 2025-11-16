from pathlib import Path


# Random state used for entire project
SEED = 42
## Absolute base path to project root directory
BASE_PATH = Path("")
# Device used for torch
# import torch
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" ## mps used for tuning ONLY (if want to use model in interface)
DEVICE = "cpu"  # cpu used for everything else
