import torch
import tacoreader.v1 as tacoreader
import rasterio as rio
import numpy as np
import torch.nn.functional as F

class TacoDataLoader(torch.utils.data.Dataset):
    def __init__(self, taco_file, size=(192, 192)):
        self.dataset = tacoreader.load(taco_file)
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.read(idx)

        s2l1c_path = sample.read(0)
        target_path = sample.read(1)

        with rio.open(s2l1c_path) as src:
            # Sentinel-2 RGB is B04, B03, B02.
            image = src.read([4, 3, 2]).astype(np.float32)

        with rio.open(target_path) as dst:
            mask = dst.read(1).astype(np.float32)

        image = image / 10000.0
        image = np.clip(image, 0, 1)

        # Binary cloud mask for BCE + Dice loss.
        mask = (mask == 1).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        image = F.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=self.size,
            mode="nearest",
        ).squeeze(0)

        return image, mask


def get_data_loaders(taco_file="mini.taco", batch_size=4):
    dataset = TacoDataLoader(taco_file)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
    }
