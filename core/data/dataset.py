import torch
import numpy as np
import pandas as pd
from typing import Callable, Dict
from pathlib import Path
from torchvision import transforms

from core.data.utils import open_tensor, generate_dataframe


class SparseTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tensor_dir: Path,
        LABEL_MAPPING: Dict[str, int],
        training: bool = True,
        transform: Callable = None,
        parse_dim: bool = False,
        fmt: str = 'pkl'
    ):
        self.tensor_dir = tensor_dir
        self.training = training
        self.transform = transform
        self.fmt = fmt

        self.df = generate_dataframe(self.tensor_dir, parse_dim, self.fmt)
        self.df['label'] = self.df['lesion_subtype'].apply(lambda lst: LABEL_MAPPING[lst])

    def __getitem__(self, index: int):
        row = self.df.loc[index]
        tensor_fp = row.tensor_path
        t = open_tensor(tensor_fp)
        if self.transform:
            t = self.transform(t)
        label = np.array([row.label]).astype(float) if self.training else np.array([-1])
        return index, t, label

    def __len__(self):
        return len(self.df)