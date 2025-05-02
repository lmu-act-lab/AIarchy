import os
import random
import numpy as np
import torch


def set_seed(SEED: int = 42):
    # 2. Python built‑in RNG
    random.seed(SEED)

    # 3. NumPy
    np.random.seed(SEED)

    # 4. PyTorch (CPU + GPU)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # enforce deterministic CuDNN (may incur a performance penalty)
    torch.use_deterministic_algorithms(True)

    # 6. Environment variable for hash seeds
    os.environ["PYTHONHASHSEED"] = str(SEED)