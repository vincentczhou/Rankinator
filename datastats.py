from pathlib import Path

import numpy as np
import hydra
import torch
from omegaconf import DictConfig

from libs import (
    get_tokenizer,
    get_dataloaders,
)
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    tokenizer = get_tokenizer(args)
    train_dataloader, val_dataloader = get_dataloaders(tokenizer, args)

    ranked = 0
    unranked = 0
    for i, data in enumerate(train_dataloader):
        s = data["labels"].sum()
        ranked += s
        unranked += torch.numel(data["labels"]) - s
    print(f"# Ranked Windows: {ranked}")
    print(f"# Unranked Windows: {unranked}")


if __name__ == "__main__":
    main()
