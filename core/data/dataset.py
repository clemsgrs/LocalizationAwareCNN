import torch
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable, Dict
from pathlib import Path
from functools import lru_cache
from torchvision import transforms

from core.data.utils import load_slide_tensor

@lru_cache(maxsize=256)
def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


class MILImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        tile_size: int = 224,
        training: bool = True,
        transform: Callable = None
    ):
        self.dataset = dataset
        self.tile_size = tile_size
        self.training = training
        self.transform = transform

    @lru_cache(maxsize=4096)
    def __getitem__(self, index: int):
        row = self.dataset.loc[index]
        slide_fp, coords = row.slide_path, row.coords
        slide = openslide.OpenSlide(slide_fp)
        tile = slide.read_region(coords, 0, (self.tile_size,self.tile_size)).convert('RGB')
        # TODO: find a way to apply the same transform to the WHOLE slide, then extract tile image
        # current solution would require to load the full slide with openslide which takes way too long...
        if self.transform:
            tile = self.transform(tile)
        else:
            tile = transforms.functional.to_tensor(tile)
        label = np.array([row.label]).astype(float) if self.training else np.array([-1])
        return index, tile, label

    def __len__(self):
        return len(self.dataset)


class SparseTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        embedding_dir: Path,
        label_mapping: Dict,
        training: bool = True,
    ):
        self.dataset = dataset
        self.embedding_dir = embedding_dir
        self.label_mapping = label_mapping
        self.training = training

    def __getitem__(self, index: int):
        row = self.dataset.loc[index]
        fp = Path(self.embedding_dir, f'{row.slide_id}.pkl')
        sparse_tensor = load_slide_tensor(fp)
        label = self.label_mapping[row.lesion_subtype]
        label = np.array([label]).astype(float) if self.training else np.array([-1])
        return index, sparse_tensor, label

    def __len__(self):
        return len(self.dataset)