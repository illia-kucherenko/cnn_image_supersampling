import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESPCN(upscale_factor=3, num_channels=1).to(device)

checkpoint_path = Path("checkpoints/model_experimental.pth")
if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    exit()

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"Model loaded on {device}\n")

diagnose_dir = Path("diagnose_experimental")
diagnose_dir.mkdir(parents=True, exist_ok=True)

test_images = [
    "resources/datasets/DIV2K/DIV2K_train_LR_bicubic/X3/0801x3.png",
    "resources/datasets/DIV2K/DIV2K_train_LR_bicubic/X3/0802x3.png",
    "resources/datasets/DIV2K/DIV2K_train_LR_bicubic/X3/0803x3.png",
]

for idx, test_lr_path in enumerate(test_images):
    if not Path(test_lr_path).exists():
        print(f"Test image not found: {test_lr_path}")
        continue

    print(f"{'='*60}\nTesting image {idx+1}: {Path(test_lr_path).name}\n{'='*60}")

    lr_img_rgb = Image.open(test_lr_path).convert("RGB")
    lr_ycbcr = lr_img_rgb.convert("YCbCr")
    lr_y, lr_cb, lr_cr = lr_ycbcr.split()

    print(f"Input (LR) size: {lr_img_rgb.size} pixels")

    lr_array = np.array(lr_y, dtype=np.float32) / 255.0
    lr_tensor = torch.from_numpy(lr_array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(lr_tensor)

    output_array = np.clip(output.squeeze().cpu().numpy(), 0, 1)
    output_uint8 = (output_array * 255).astype(np.uint8)
    sr_y = Image.fromarray(output_uint8)

    sr_cb = lr_cb.resize(sr_y.size, Image.Resampling.BICUBIC)
    sr_cr = lr_cr.resize(sr_y.size, Image.Resampling.BICUBIC)

    upscaled_img = Image.merge("YCbCr", (sr_y, sr_cb, sr_cr)).convert("RGB")
    output_rgb_array = np.array(upscaled_img, dtype=np.float32) / 255.0

    print(f"Output (SR) size: {upscaled_img.size} pixels")
    print(f"Expected size (3x): {(lr_img_rgb.size[0]*3, lr_img_rgb.size[1]*3)} pixels")
    print(f"Size match: {upscaled_img.size == (lr_img_rgb.size[0]*3, lr_img_rgb.size[1]*3)}")

    bicubic_resized = lr_img_rgb.resize((lr_img_rgb.size[0]*3, lr_img_rgb.size[1]*3), Image.Resampling.BICUBIC)
    bicubic_array = np.array(bicubic_resized, dtype=np.float32) / 255.0

    hr_path = test_lr_path.replace("DIV2K_train_LR_bicubic/X3/", "DIV2K_train_HR/").replace("x3.png", ".png")
    if Path(hr_path).exists():
        hr_img_rgb = Image.open(hr_path).convert("RGB")
        hr_array = np.array(hr_img_rgb, dtype=np.float32) / 255.0

        if hr_array.shape[:2] != output_rgb_array.shape[:2]:
            min_h = min(hr_array.shape[0], output_rgb_array.shape[0])
            min_w = min(hr_array.shape[1], output_rgb_array.shape[1])
            hr_array = hr_array[:min_h, :min_w, :]
            output_rgb_array_cropped = output_rgb_array[:min_h, :min_w, :]
        else:
            output_rgb_array_cropped = output_rgb_array

        if bicubic_array.shape[:2] != hr_array.shape[:2]:
            min_h = min(hr_array.shape[0], bicubic_array.shape[0])
            min_w = min(hr_array.shape[1], bicubic_array.shape[1])
            hr_array_crop = hr_array[:min_h, :min_w, :]
            bicubic_array = bicubic_array[:min_h, :min_w, :]
        else:
            hr_array_crop = hr_array

        espcn_psnr = psnr(hr_array_crop, output_rgb_array_cropped, data_range=1.0)
        espcn_ssim = ssim(hr_array_crop, output_rgb_array_cropped, data_range=1.0, channel_axis=-1)

        bicubic_psnr = psnr(hr_array_crop, bicubic_array, data_range=1.0)
        bicubic_ssim = ssim(hr_array_crop, bicubic_array, data_range=1.0, channel_axis=-1)

        print(f"\nQuality Metrics (RGB):")
        print(f"  ESPCN  - PSNR: {espcn_psnr:.2f}dB, SSIM: {espcn_ssim:.4f}")
        print(f"  Bicubic - PSNR: {bicubic_psnr:.2f}dB, SSIM: {bicubic_ssim:.4f}")
        print(f"  Improvement: PSNR +{espcn_psnr - bicubic_psnr:.2f}dB, SSIM +{espcn_ssim - bicubic_ssim:.4f}")
    else:
        print(f"HR reference not found: {hr_path}")

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    axes[0].imshow(np.array(lr_img_rgb))
    axes[0].set_title(f"Low-Res Input\n{lr_img_rgb.size[0]}×{lr_img_rgb.size[1]}px")
    axes[0].axis('off')

    axes[1].imshow(output_rgb_array)
    axes[1].set_title(f"ESPCN Output\n{upscaled_img.size[0]}×{upscaled_img.size[1]}px")
    axes[1].axis('off')

    axes[2].imshow(bicubic_array)
    axes[2].set_title(f"Bicubic Baseline\n{bicubic_resized.size[0]}×{bicubic_resized.size[1]}px")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(diagnose_dir / f"diagnosis_image_{idx+1}.png", dpi=300)
    upscaled_img.save(diagnose_dir / f"test_output_{idx+1}.png")
    print(f"Saved diagnosis to diagnose_experimental/diagnosis_image_{idx+1}.png")
    print(f"Saved upscaled to diagnose_experimental/test_output_{idx+1}.png")

print("\nDone!")
