import timm
import torch
import pickle
import openslide
import pandas as pd
from typing import Union
from pathlib import Path

from core.data.extract_tissue import make_sample_grid


def slide_dataframe(slide_dir: Path, fmt: str = 'svs'):
    slide_dimensions, slide_ids = [], []
    slide_paths, lesion_types, lesion_subtypes = [], [], []
    lesion_types_dir = [sd for sd in slide_dir.iterdir() if sd.is_dir()]
    for lt in lesion_types_dir:
        lesion_subtypes_dir = [sd for sd in lt.iterdir() if sd.is_dir()]
        for lst in lesion_subtypes_dir:
            slide_filenames = list(Path(lst).glob(f'*.{fmt}'))
            for fp in slide_filenames:
                slide_paths.append(str(fp))
                slide = openslide.OpenSlide(str(fp))
                d = slide.dimensions
                slide_dimensions.append(d)
                slide_id = Path(fp).stem
                slide_ids.append(slide_id)
                lesion_types.append(lt.stem)
                lesion_subtypes.append(lst.stem)
    slide_dict = {
        'slide_id': slide_ids,
        'slide_path': slide_paths,
        'slide_dimension': slide_dimensions,
        'lesion_type': lesion_types,
        'lesion_subtype': lesion_subtypes
    }
    df = pd.DataFrame(slide_dict)
    return df


def get_valid_tiles(row: pd.Series, tile_size: int = 224, filter_tissue: bool = False):
    if isinstance(row.slide_dimension, tuple):
        max_y, max_x = row.slide_dimension
    elif isinstance(row.slide_dimension, str):
        max_y, max_x = list(map(int, row.slide_dimension[1:-1].split(',')))
    # get coords of tissue patches in whole slide images using Otsu algorithm
    if filter_tissue:
        slide = openslide.OpenSlide(row.slide_path)
        coords, _ = make_sample_grid(slide, tile_size, 20, 10, 10, False, prune=False, overlap=0)
    # get coordinates of all patches in whole slide images
    else:
        coords = []
        for y in range(0, max_y, tile_size):
            for x in range(0, max_x, tile_size):
                coords.append((y,x))
    return coords


def tiles_dataframe(df: pd.DataFrame, tile_size: int = 224, filter_tissue: bool = False):
    df['tiles'] = df.apply(get_valid_tiles, axis=1, tile_size=tile_size, filter_tissue=filter_tissue)
    df = df.explode('tiles').rename(columns={'tiles': 'coords'})
    df = df.reset_index(drop=True)
    return df


def generate_embeddings(input, arc: str = 'seresnet50', verbose: bool = False):
    model = timm.create_model(arc, pretrained=True, num_classes=0)
    with torch.no_grad():
        emb = model(input)
    if verbose:
        print(f'emb.shape: {emb.shape}')
    return emb


def save_slide_tensor(filepath, t):
    with open(filepath, 'wb') as f:
        pickle.dump(t, f)


def load_slide_tensor(filepath):
    with open(filepath, 'rb') as f:
        t = pickle.load(f)
    return t