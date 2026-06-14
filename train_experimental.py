import torch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = False

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss, L1Loss
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=64, upscale_factor=2, img_range=None, patches_per_epoch=50):
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size - (patch_size % upscale_factor)
        self.patches_per_epoch = patches_per_epoch

        hr_files = sorted(Path(hr_dir).glob('*.png'))
        lr_files = sorted(Path(lr_dir).glob('*.png'))

        if len(hr_files) != len(lr_files):
            raise ValueError(f"Mismatch: {len(hr_files)} HR images vs {len(lr_files)} LR images")

        if img_range:
            start, end = img_range
            def get_image_id(filepath):
                import re
                match = re.match(r'(\d+)', filepath.stem)
                return int(match.group(1)) if match else -1

            hr_files = [f for f in hr_files if start <= get_image_id(f) <= end]
            lr_files = [f for f in lr_files if start <= get_image_id(f) <= end]

        print(f"Preloading {len(hr_files)} images into memory to speed up training...")
        self.hr_images = []
        self.lr_images = []

        for hr_path, lr_path in zip(hr_files, lr_files):
            hr_img = Image.open(hr_path).convert('YCbCr').split()[0]
            lr_img = Image.open(lr_path).convert('YCbCr').split()[0]

            self.hr_images.append(torch.from_numpy(np.array(hr_img)))
            self.lr_images.append(torch.from_numpy(np.array(lr_img)))

            hr_img.close()
            lr_img.close()

        print(f"Dataset ready. Will extract {self.patches_per_epoch} random patches per image per epoch.")

    def __len__(self):
        return len(self.hr_images) * self.patches_per_epoch

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_epoch

        hr_img_t = self.hr_images[img_idx]
        lr_img_t = self.lr_images[img_idx]

        hr_h, hr_w = hr_img_t.shape

        patch_size = (self.patch_size // self.upscale_factor) * self.upscale_factor
        lr_patch_size = patch_size // self.upscale_factor

        max_x = hr_w - patch_size
        max_y = hr_h - patch_size

        if max_x <= 0 or max_y <= 0:
            x, y = 0, 0
        else:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            x -= x % self.upscale_factor
            y -= y % self.upscale_factor

        lr_x = x // self.upscale_factor
        lr_y = y // self.upscale_factor

        hr_patch = hr_img_t[y : y + patch_size, x : x + patch_size]
        lr_patch = lr_img_t[lr_y : lr_y + lr_patch_size, lr_x : lr_x + lr_patch_size]

        if hr_patch.shape[0] < patch_size or hr_patch.shape[1] < patch_size:
            pad_h = max(0, patch_size - hr_patch.shape[0])
            pad_w = max(0, patch_size - hr_patch.shape[1])
            hr_patch = torch.nn.functional.pad(hr_patch, (0, pad_w, 0, pad_h), mode='reflect')

        if lr_patch.shape[0] < lr_patch_size or lr_patch.shape[1] < lr_patch_size:
            pad_h = max(0, lr_patch_size - lr_patch.shape[0])
            pad_w = max(0, lr_patch_size - lr_patch.shape[1])
            lr_patch = torch.nn.functional.pad(lr_patch, (0, pad_w, 0, pad_h), mode='reflect')

        if random.random() < 0.5:
            hr_patch = torch.flip(hr_patch, [1])
            lr_patch = torch.flip(lr_patch, [1])

        if random.random() < 0.5:
            hr_patch = torch.flip(hr_patch, [0])
            lr_patch = torch.flip(lr_patch, [0])

        hr_tensor = hr_patch.float().unsqueeze(0) / 255.0
        lr_tensor = lr_patch.float().unsqueeze(0) / 255.0

        return lr_tensor, hr_tensor

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_loader, desc="Training")

    for low_res, high_res in progress_bar:
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        sr_output = model(low_res)
        loss = criterion(sr_output, high_res)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            sr_output = model(low_res)
            loss = criterion(sr_output, high_res)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=50, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'model_experimental.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved (Val Loss: {val_loss:.6f}) to {checkpoint_path}")

    return train_losses, val_losses


def plot_training_history(train_losses, val_losses, filename='training_history_experimental.png', data_filename='training_history_experimental.txt'):
    with open(data_filename, 'w') as f:
        f.write("Epoch,Train Loss,Validation Loss\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")

    plt.figure(figsize=(12, 7))
    epochs = np.arange(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2, markersize=6)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (L1)', fontsize=12)
    plt.title('ESPCN Experimental Training History (3x)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    print(f"Training history saved to {filename}")
    print(f"Training data saved to {data_filename}")
    plt.close()


if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    upscale_factor = 3
    num_epochs = 50
    learning_rate = 1e-4
    batch_size = 16

    train_hr_dir = "resources/datasets/DIV2K/DIV2K_train_HR"
    train_lr_dir = "resources/datasets/DIV2K/DIV2K_train_LR_bicubic/X3"
    valid_hr_dir = "resources/datasets/DIV2K/DIV2K_train_HR"
    valid_lr_dir = "resources/datasets/DIV2K/DIV2K_train_LR_bicubic/X3"

    print("\nLoading training dataset...")
    train_dataset = SRDataset(train_hr_dir, train_lr_dir, patch_size=64, upscale_factor=upscale_factor, img_range=(1, 800), patches_per_epoch=50)

    print("\nLoading validation dataset...")
    valid_dataset = SRDataset(valid_hr_dir, valid_lr_dir, patch_size=64, upscale_factor=upscale_factor, img_range=(801, 900), patches_per_epoch=5)

    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")
    print(f"Using {num_workers} worker threads for data loading")

    model = ESPCN(upscale_factor=upscale_factor, num_channels=1, hidden_channels=128).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

    criterion = L1Loss()

    print("Starting training...")
    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs)

    try:
        plot_training_history(train_losses, val_losses)
    except Exception as e:
        print(f"Could not save training history plot: {e}")
