import tqdm
import timm
import time
import torch
import pickle
import istarmap
import openslide
import numpy as np
import multiprocessing as mp

from itertools import repeat
from pathlib import Path
from torchvision import transforms

from core.data.utils import slide_dataframe, tiles_dataframe, save_slide_tensor


def generate_embedding(model, slide, coord, tile_size: int = 224):
    tile = slide.read_region(coord, 0, (tile_size,tile_size)).convert('RGB')
    tile = transforms.functional.to_tensor(tile).cuda()
    # add batch dimension
    tile = tile.unsqueeze(0)
    with torch.no_grad():
        emb = model(tile)
    return emb


if __name__ == '__main__':

    N = 200
    tile_size = 224
    filter_tissue = True
    tensor_dir = Path('./tensors/train')

    data_dir = Path('/Users/clementgrisi/Code/datasets/bracs/train')
    slide_df = slide_dataframe(data_dir)
    slide_df.to_csv('data/slides_train.csv', index=False)
    print(f'#slides: {len(slide_df)}')

    tiles_df = tiles_dataframe(slide_df, tile_size, filter_tissue)
    tiles_df.to_csv('data/tiles_train.csv', index=False)
    print(f'#tiles: {len(tiles_df)}')

    # these coordinates are in the openslide referential (x <> width, y <> height)
    # be careful how they're used in following part of code as axis get transposed
    # when turning a PIL image into an array or a tensor
    tiles_df.loc[:,'x'] = tiles_df.coords.map(lambda x: x[0])
    tiles_df.loc[:,'y'] = tiles_df.coords.map(lambda x: x[1])

    min_size = 32

    model = timm.create_model('seresnet50', pretrained=True, num_classes=0)
    model = model.cuda()

    for slide_id, gdf in tiles_df.groupby('slide_id'):
  
        slide_fp = gdf.slide_path.iloc[0]  
        slide = openslide.OpenSlide(slide_fp)

        coords = gdf.coords.values.tolist()

        max_x, max_y = tiles_df.x.max(), tiles_df.y.max()
        min_x, min_y = tiles_df.x.min(), tiles_df.y.min()
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        M = max(delta_x,delta_y)
        m = int(np.ceil(M/tile_size))

        if m < min_size:
            m = min_size

        emb_size = model.layer4[-1].conv3.out_channels
        sparse_tensor = torch.zeros((m,m,emb_size))

        with tqdm.tqdm(coords,
                    desc=(f'{slide_id}'),
                    unit=' tile',
                    ncols=100
        ) as t:

            # (x,y) still in openslide referential
            for (x,y) in t:

                e = generate_embedding(model, slide, (x,y))
                i, j = y // tile_size, x // tile_size
                sparse_tensor[i,j] = e.cpu()

            # save (sparse) tensor to disk
            lesion_type = gdf.lesion_type.iloc[0]
            lesion_subtype = gdf.lesion_subtype.iloc[0]
            save_path = Path(tensor_dir, lesion_type, lesion_subtype, f'{slide_id}.pkl')
            save_dir = save_path.parents[0]
            save_dir.mkdir(exist_ok=True, parents=True)
            save_slide_tensor(save_path, sparse_tensor)
        