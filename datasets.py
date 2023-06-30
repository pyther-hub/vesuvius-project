import albumentations as A
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import *


def get_transforms(data, cfg):
    if data == "train":
        aug = A.Compose(cfg.train_aug_list)
    elif data == "valid":
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data["image"]
            label = data["mask"]

        return image, label


def make_test_dataset(fragment_id):
    test_images = read_image(fragment_id)

    x1_list = list(range(0, test_images.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, test_images.shape[0] - CFG.tile_size + 1, CFG.stride))

    test_images_list = []
    xyxys = []
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size

            test_images_list.append(test_images[y1:y2, x1:x2])
            xyxys.append((x1, y1, x2, y2))
    xyxys = np.stack(xyxys)

    test_dataset = CustomDataset(
        test_images_list, CFG, transform=get_transforms(data="valid", cfg=CFG)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, xyxys


def get_train_valid_dataset(valid_id: int = 1):
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in [1, 2, 3]:
        image, mask = read_image_mask(fragment_id)

        x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, CFG.stride))
        y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, CFG.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size

                if fragment_id == valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def read_image(fragment_id):
    images = []

    start = CFG.start_chans
    end = CFG.end_chans
    idxs = range(start, end)

    for i in tqdm(idxs):
        image = cv2.imread(
            CFG.comp_dataset_path + f"test/{fragment_id}/surface_volume/{i:02}.tif", 0
        )

        pad0 = CFG.tile_size - image.shape[0] % CFG.tile_size
        pad1 = CFG.tile_size - image.shape[1] % CFG.tile_size

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)
    return images


def read_image_mask(fragment_id):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        image = cv2.imread(
            CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0
        )

        pad0 = CFG.tile_size - image.shape[0] % CFG.tile_size
        pad1 = CFG.tile_size - image.shape[1] % CFG.tile_size

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype("float32")
    mask /= 255.0

    return images, mask


def get_data_loader(valid_id, verbose=False):
    (
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        valid_xyxys,
    ) = get_train_valid_dataset(valid_id=valid_id)
    valid_xyxys = np.stack(valid_xyxys)

    if verbose:
        print("DATASET ", valid_id)
    train_dataset = CustomDataset(
        train_images,
        CFG,
        labels=train_masks,
        transform=get_transforms(data="train", cfg=CFG),
    )
    valid_dataset = CustomDataset(
        valid_images,
        CFG,
        labels=valid_masks,
        transform=get_transforms(data="valid", cfg=CFG),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader, valid_xyxys
