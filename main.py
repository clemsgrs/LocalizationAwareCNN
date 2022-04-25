import torch
import pickle
import openslide
import numpy as np
from pathlib import Path
from torchvision import transforms

from core.data.utils import slide_dataframe, tiles_dataframe, generate_embeddings, save_slide_tensor


if __name__ == '__main__':
    
    tile_size = 224
    filter_tissue = True
    embedding_dir = Path('./embeddings/train')
    
    slide_dir = Path('./slides/train')
    slide_df = slide_dataframe(slide_dir)
    # slide_df.to_csv('slide.csv', index=False)
    print(f'#slides: {len(slide_df)}')


    tiles_df = tiles_dataframe(slide_df, tile_size, filter_tissue)
    # tiles_df.to_csv('tiles.csv', index=False)
    print(f'#tiles: {len(tiles_df)}')

    # from torch.utils.data import DataLoader
    # from core.data.dataset import MILImageDataset

    # dataset = MILImageDataset(tiles_df, tile_size=tile_size, training=False)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # index, tile, label = next(iter(dataloader))

    # h, w = tiles_df.loc[index].slide_dimensions.values[0]
    # m = int(np.ceil(max(h,w)/tile_size))
    # t = torch.zeros((m,m,2048))
    # print(f't.shape: {t.shape}')

    # y, x = tiles_df.loc[index].coords.values[0]
    # print(f'{y}, {x}')
    # small_y, small_x = y // tile_size, x // tile_size
    # print(f'{small_y}, {small_x}')

    # emb = generate_embeddings(tile)
    # t[small_y,small_x] = emb

    # # save
    # slide_id = tiles_df.loc[index].slide_id.values[0]
    # with open(f'{slide_id}.pkl', 'wb') as f:
    #     pickle.dump(t, f)
    
    tiles_df.loc[:,'x'] = tiles_df.coords.map(lambda x: x[1])
    tiles_df.loc[:,'y'] = tiles_df.coords.map(lambda x: x[0])

    # generate fixed size inputs for incomming CNN training
    max_x, max_y = tiles_df.x.max(), tiles_df.y.max()
    min_x, min_y = tiles_df.x.min(), tiles_df.y.min()
    mx = max_x - min_y
    my = max_y - min_y
    M = max(mx,my)
    print(f'mx: {mx} ; my: {my} ; M: {M}')
    m = int(np.ceil(M/224))
    print(f'm: {m}')

    print()
    for slide_id, gdf in tiles_df.groupby('slide_id'):
        print(f'slide_id: {slide_id}')
        slide_fp = gdf.slide_path.iloc[0]
        slide = openslide.OpenSlide(slide_fp)
        t = torch.zeros((m,m,2048))
        for i, (y,x) in enumerate(gdf.coords):
            print(f'\rprocessing tile {i+1}/{len(gdf)}', end='')
            tile = slide.read_region((y,x), 0, (tile_size,tile_size)).convert('RGB')
            tile = transforms.functional.to_tensor(tile)
            # add batch dimension
            tile = tile.unsqueeze(0)
            emb = generate_embeddings(tile)
            i, j = y // tile_size, x // tile_size
            t[i,j] = emb
        # save (sparse) tensor to disk
        save_path = Path(embedding_dir, f'{slide_id}.pkl')
        save_slide_tensor(save_path, t)        