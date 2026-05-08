import numpy as np
import rasterio as rio
import tacoreader.v1 as tacoreader
from PIL import Image
import random


TACO_FILE = "../datasets/cloudsen12/mini.taco"
OUTPUT_FILE = "calib_set.npy"

IMAGE_SIZE = (512, 512)
NUM_IMAGES = 128


def read_rgb_from_sample(sample):
    s2l1c_path = sample.read(0)

    with rio.open(s2l1c_path) as src:
        # Sentinel-2 RGB bands: B04, B03, B02
        image = src.read([4, 3, 2]).astype(np.float32)

    # Same scaling used during training
    image = image / 10000.0
    image = np.clip(image, 0, 1)

    # CHW -> HWC
    image = np.transpose(image, (1, 2, 0))

    # Resize to model input size
    image = Image.fromarray((image * 255).astype(np.uint8))
    image = image.resize(IMAGE_SIZE)

    image = np.asarray(image).astype(np.float32)

    return image


def main():
    dataset = tacoreader.load(TACO_FILE)

    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)

    images = []

    for idx in indices[:NUM_IMAGES]:
        sample = dataset.read(idx)
        image = read_rgb_from_sample(sample)
        images.append(image)

    calib_set = np.stack(images, axis=0)

    print("Calibration set shape:", calib_set.shape)
    print("Calibration set dtype:", calib_set.dtype)
    print("Min:", calib_set.min(), "Max:", calib_set.max())

    np.save(OUTPUT_FILE, calib_set)

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
