import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    comp_dir_path = "/kaggle/input/"
    comp_folder_name = "vesuvius-challenge-ink-detection"
    comp_dataset_path = f"{comp_dir_path}{comp_folder_name}/"
    log_path = f"output/logs/{comp_folder_name}/"
    log_dir=log_path
    submission_dir="output/submissions"
    model_dir="output/models"

    # ============== pred target =============
    target_size = 1
    TTA = True

    # ============== model cfg =============
    model_name = "Unet"
    backbone = "mit_b3"
    start_chans = 28
    end_chans = 31
    #     chans_to_choose=[27,28,29]
    in_chans = 3  # end_chans-start_chans
    # ============== training cfg =============
    size = 224
    tile_size = 224
    stride = tile_size // 2
    #     in_chans=end_chans-start_chans

    batch_size = 16  # 32
    use_amp = True

    scheduler = "GradualWarmupSchedulerV2"
    # scheduler = 'CosineAnnealingLR'
    epochs = 15

    warmup_factor = 10
    lr = 1e-4 / warmup_factor

    # ============== fold =============

    metric_direction = "maximize"

    # ============== fixed =============
    pretrained = True
    inf_weight = "best"

    min_lr = 1e-6

    num_workers = 2

    seed = 42

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf(
            [
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_width=int(size * 0.3),
            max_height=int(size * 0.3),
            mask_fill_value=0,
            p=0.5,
        ),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]
