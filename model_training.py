#
# This is an attempt to have Edge Aerospace own's Unet code.
# 
#

import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import toml

from unet import UNet

# Taco dataset related
import rasterio as rio
import tacoreader
from torch.utils.data import random_split
from taco_data_loader import TacoDataLoader

import argparse

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def get_data_loaders(taco_file="datasets/cloudsen12/mini.taco", batch_size=4):
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

def train_model(model, optimizer, scheduler, num_epochs=60,batch_size=25):
    dataloaders = get_data_loaders(batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    save_validation_prediction(model, dataloaders, "val_prediction.png")
    return model

def save_validation_prediction(model, dataloaders, output_path="val_prediction.png"):
    """
    Prediction Image generated to show model segmentation performance
    this is how the combined image file should be interpreted: 
    [input image] [ground truth] [soft probability] [binary segmentation]
    """
    from PIL import Image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    inputs, labels = next(iter(dataloaders["val"]))
    inputs = inputs.to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(inputs))

    img = inputs[0].detach().cpu().numpy()
    gt = labels[0].detach().cpu().numpy()[0]
    mask = pred[0].detach().cpu().numpy()[0]

    # probability map
    prob = (mask * 255).astype(np.uint8)

    # binary threshold map
    binary = ((mask > 0.5) * 255).astype(np.uint8)

    # RGB conversions for visualization
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)

    gt = np.stack([gt, gt, gt], axis=-1)

    prob_rgb = np.stack([prob, prob, prob], axis=-1)
    binary_rgb = np.stack([binary, binary, binary], axis=-1)

    combined = np.concatenate([
        (img * 255).astype(np.uint8),
        (gt * 255).astype(np.uint8),
        prob_rgb,
        binary_rgb
    ], axis=1)

    Image.fromarray(combined).save(output_path)
    print(f"Saved validation comparison to {output_path}")

def run(UNet, num_epochs=60, batch_size=25):
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft,
        step_size=30,
        gamma=0.1
    )

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs,batch_size)

    torch.save(model.state_dict(), "unet_segmentation.pth")
    print("#==================== Results ====================#")
    print("Saved PyTorch weights to unet_segmentation.pth")

    model.eval()

    trans = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder("val", transform=trans)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)

    labels = torch.zeros(
        inputs.size(0),
        num_class,
        inputs.size(2),
        inputs.size(3)
    ).to(device)

    with torch.no_grad():
        pred = model(inputs)
        pred = torch.sigmoid(pred)

    pred = pred.data.cpu().numpy()

    # Load test image for onnx export
    from PIL import Image
    img = Image.open("test_img.png").convert("RGB")
    input_tensor = trans(img).unsqueeze(0).to(device)

    # Save model in onnx format(important for Hailo dfc conversion later)
    torch.onnx.export(
    model,
    input_tensor,
    "unet_segmentation.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
    )

    print("Saved ONNX model to unet_imagenette.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.config is not None:
        config = toml.load(args.config)

        args.lr = config.get("lr", args.lr)
        args.epochs = config.get("epochs", args.epochs)
        args.batch_size = config.get("batch_size", args.batch_size)

    run(UNet,num_epochs=args.epochs,batch_size=args.batch_size)