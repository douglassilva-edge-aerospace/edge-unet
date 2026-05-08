import os
import numpy as np
import rasterio as rio
import tacoreader.v1 as tacoreader
from PIL import Image

TACO_FILE = "datasets/cloudsen12/mini.taco"
OUT_DIR = "val"
IMAGE_SIZE = (512, 512)
NUM_SAMPLES = 15

os.makedirs(OUT_DIR, exist_ok=True)

dataset = tacoreader.load(TACO_FILE)

for idx in range(5,min(NUM_SAMPLES, len(dataset))):
    sample = dataset.read(idx)

    s2l1c_path = sample.read(0)
    mask_path = sample.read(1)

    with rio.open(s2l1c_path) as src:
        image = src.read([4, 3, 2]).astype(np.float32)

    with rio.open(mask_path) as dst:
        mask = dst.read(1).astype(np.float32)

    # Same Sentinel scaling
    image = image / 10000.0
    image = np.clip(image, 0, 1)

    # CHW -> HWC
    image = np.transpose(image, (1, 2, 0))

    # Resize
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_pil = image_pil.resize(IMAGE_SIZE)

    mask = (mask == 1).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.resize(IMAGE_SIZE)

    image_uint8 = np.asarray(image_pil).astype(np.uint8)

    np.save(os.path.join(OUT_DIR, f"sample_{idx}.npy"), image_uint8)
    image_pil.save(os.path.join(OUT_DIR, f"sample_{idx}_rgb.png"))
    mask_pil.save(os.path.join(OUT_DIR, f"sample_{idx}_gt.png"))

    print("saved", idx, image_uint8.shape, image_uint8.min(), image_uint8.max())