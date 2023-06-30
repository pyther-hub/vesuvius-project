import albumentations as A
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import gc
import cv2
from utils import *
from config import CFG
import segmentation_models_pytorch as smp


DiceLoss = smp.losses.DiceLoss(mode="binary")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()


def criterion(y_pred, y_true):
    #     return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)
    #     return BCELoss(y_pred, y_true)
    return BCELoss(y_pred, y_true) + DiceLoss(y_pred, y_true)


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

    return losses.avg
