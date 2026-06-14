import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import time
import os


class ESPCN(nn.Module):
    def __init__(self, upscale_factor=3, num_channels=1, hidden_channels=128):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.hidden_channels = hidden_channels

        self.conv1 = nn.Conv2d(num_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(hidden_channels, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        res1 = x
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x + res1
        res2 = x
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x + res2
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        return x


def upscale_image(model, img_path, device):
    try:
        img_rgb = Image.open(img_path).convert("RGB")
        original_size = img_rgb.size

        img_ycbcr = img_rgb.convert("YCbCr")
        lr_y, lr_cb, lr_cr = img_ycbcr.split()

        img_array = np.array(lr_y, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            output = model(img_tensor)
        inference_time = time.time() - start_time

        output_array = np.clip(output.squeeze().cpu().numpy(), 0, 1)
        output_uint8 = (output_array * 255).astype(np.uint8)
        sr_y = Image.fromarray(output_uint8)

        sr_cb = lr_cb.resize(sr_y.size, Image.Resampling.BICUBIC)
        sr_cr = lr_cr.resize(sr_y.size, Image.Resampling.BICUBIC)

        upscaled_img = Image.merge("YCbCr", (sr_y, sr_cb, sr_cr)).convert("RGB")

        return upscaled_img, original_size, inference_time
    except Exception as e:
        print(f"   Error processing {img_path}: {e}")
        return None, None, None


def process_images_folder(model, folder_path, device):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return

    print(f"\nProcessing images in {folder_path}")
    print("=" * 70)

    results_dir = folder_path / "upscaled"
    results_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        image_files.extend(list(folder_path.glob(ext)))
        image_files.extend(list(folder_path.glob(ext.upper())))

    image_files = list(set([f for f in image_files if "upscaled" not in f.parts]))

    if not image_files:
        print(f"No images found in {folder_path}")
        return

    times = []

    for idx, img_path in enumerate(sorted(image_files), 1):
        upscaled, orig_size, inf_time = upscale_image(model, img_path, device)

        if upscaled is None:
            continue

        times.append(inf_time)

        print(f"{idx:2d}. {img_path.name:20s} | Size: {orig_size} -> {upscaled.size} | Time: {inf_time:.3f}s")

        output_path = results_dir / f"{img_path.stem}_upscaled.png"
        upscaled.save(output_path)

    if times:
        print("-" * 70)
        print(f"Average inference time: {np.mean(times):.3f}s +/- {np.std(times):.3f}s")
        print(f"Results saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ESPCN Inference Test")
    print(f"Device: {device}\n")

    checkpoint_path = Path("checkpoints/model_experimental.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for cp in Path("checkpoints").glob("*.pth"):
            print(f"  - {cp}")
        exit()

    model = ESPCN(upscale_factor=3, num_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded: {checkpoint_path}\n")

    process_images_folder(model, "img", device)

    print("\nProcessing complete!")
