import torch
from torchvision import transforms
from PIL import Image

from unet import UNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- model ---
    model = UNet(3).to(device)  # must match training (num_class=3)
    model.load_state_dict(torch.load("unet_imagenette.pth", map_location=device))
    model.eval()

    # --- preprocessing (same as training!) ---
    trans = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- use real image instead of dummy ---
    img = Image.open("test_img.png").convert("RGB")
    input_tensor = trans(img).unsqueeze(0).to(device)

    # --- export ---
    torch.onnx.export(
        model,
        input_tensor,
        "unet_imagenette.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print("ONNX model exported to unet_imagenette.onnx")


if __name__ == "__main__":
    main()