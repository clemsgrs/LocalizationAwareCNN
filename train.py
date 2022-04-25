import time
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

from core.data.dataset import SparseTensorDataset
from utils import open_config_file, epoch_time, run_training, run_validation

LABEL_MAPPING = {
    'n': 0,
    'pb': 0,
    'udh': 0,
    'adh': 1,
    'fea': 1,
    'ic': 1,
    'dcis': 1
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/default.json', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

train_df = pd.read_csv('train_slides.csv')
train_embedding_dir = Path(params.embedding_dir, 'train')
train_dataset = SparseTensorDataset(train_df, train_embedding_dir, LABEL_MAPPING)

val_df = pd.read_csv('val_slides.csv')
val_embedding_dir = Path(params.embedding_dir, 'val')
val_dataset = SparseTensorDataset(val_df, val_embedding_dir, LABEL_MAPPING, training=False)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=params.batch_size,
)

model = timm.create_model(params.model, pretrained=True, num_classes=params.num_classes)

optimizer = optim.Adam(model.parameters(), lr=params.lr)
if params.lr_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step, gamma=0.1)
model = model.cuda()

criterion = nn.BCEWithLogitsLoss()
criterion = criterion.cuda()

best_val_loss = float('inf')
if params.metric_objective == 'max':
    best_val_metric = 0.0
elif params.metric_objective == 'min':
    best_val_metric = float('inf')

train_losses, val_losses = [], []
train_metrics, val_metrics = [], []

for epoch in range(params.nepochs):

    start_time = time.time()

    train_loss, train_metric = run_training(
        epoch+1, 
        model, 
        train_dataset, 
        optimizer, 
        criterion, 
        params
    )
    train_losses.append(train_loss)
    train_metrics.append(train_metric)

    if epoch % params.eval_every == 0:
        
        val_loss, val_metric = run_validation(
            epoch+1, 
            model, 
            val_dataset, 
            criterion, 
            params
        )
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        if params.tracking == 'loss':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')

        else:
            val_metric_tracked = val_metrics[params.tracking]
            if val_metric_tracked > best_val_metric:
                best_val_metric = val_metric_tracked
                torch.save(model.state_dict(), 'best_model.pt')

    if params.lr_scheduler:
        scheduler.step()

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'End of epoch {epoch+1} / {params.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s')
    if params.tracking == 'loss':
        print(f'Train loss: {train_loss:.5f}')
        print(f'Val loss: {val_loss:.5f} (best Val {params.tracking}: {best_val_loss:.4f}\n')
    else:
        print(f'Train loss: {train_loss:.5f} \t Train {params.tracking}: {train_metrics[params.tracking]:.4f}')
        print(f'Val loss: {val_loss:.5f} \t Val {params.tracking}: {val_metrics[params.tracking]:.4f} (best Val {params.tracking}: {best_val_metric:.4f}\n')