import time
import cv2
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import fbeta_score
from utils import *
import wandb
from config import CFG
from datasets import *
from model import build_model
from train import train_fn
from validate import valid_fn
import gc


wandb.init(project="vesuvius-proj", config=CFG)
Logger = init_logger(log_file=CFG.log_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fragment_id in [1, 2, 3]:
    if CFG.metric_direction == "minimize":
        best_score = np.inf
    elif CFG.metric_direction == "maximize":
        best_score = -1

    best_loss = np.inf

    Logger.info(f"--------------model number:-  {fragment_id}----------------")

    train_loader, valid_loader, valid_xyxys = get_data_loader(fragment_id)

    valid_mask_gt = cv2.imread(
        CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0
    )
    valid_mask_gt = valid_mask_gt / 255

    pad0 = CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size
    pad1 = CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size

    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    model = build_model(CFG, None)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_scheduler(CFG, optimizer)

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # Train
        avg_loss = train_fn(train_loader, model, optimizer, device)

        # Evaluate
        avg_val_loss, mask_pred = valid_fn(
            valid_loader, model, device, valid_xyxys, valid_mask_gt
        )

        scheduler.step()
        best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

        elapsed = time.time() - start_time

        Logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        Logger.info(f"Epoch {epoch+1} - avgScore: {best_dice:.4f}")

        wandb.log(
            {
                "Epoch": epoch + 1,
                "avg_train_loss": avg_loss,
                "avg_val_loss": avg_val_loss,
                "time_elapsed": elapsed,
                "SCORE": best_dice,
            }
        )

        if CFG.metric_direction == "minimize":
            update_best = best_dice < best_score
        elif CFG.metric_direction == "maximize":
            update_best = best_dice > best_score

        gc_collect()

        if update_best:
            print("------UPDATE BESTTT------")
            torch.save(
                {"model": model.state_dict()},
                f"{CFG.model_dir}/{CFG.backbone}_fold_{fragment_id}_best.pth",
            )

            best_loss = avg_val_loss
            best_score = best_dice

            Logger.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            Logger.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model")

wandb.finish()
