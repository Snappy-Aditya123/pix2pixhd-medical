import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
class SegDataSet(Dataset):
    def __init__(self, base_dir, image_dir, mask_dir, type="lesion"):
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_paths = [
            os.path.join(base_dir, image_dir, f)
            for f in os.listdir(os.path.join(base_dir, image_dir))
        ]

        self.mask_paths = [
            os.path.join(base_dir, mask_dir, f)
            for f in os.listdir(os.path.join(base_dir, mask_dir))
        ]

        # sort so pairs align
        self.image_paths.sort()
        self.mask_paths.sort()

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        print("Trying to open:", img_path)
        print("Trying to open:", mask_path)
        print("Exists IMG:", os.path.exists(img_path))
        print("Exists MASK:", os.path.exists(mask_path))
        transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])

        transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),    # mask becomes float in [0,1]
    ])

        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")
        image = transform_img(image)
        mask = transform_mask(mask)


        return image, mask

    def __len__(self):
        return len(self.image_paths)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downsampling: MaxPool → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling: transpose conv → concat → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # pad x1 to match skip size
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)

# --------------------------
# LOSS FUNCTIONS
# --------------------------

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(1,2,3))
    den = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return 1 - (num / den).mean()

def iou_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum((1,2,3))
    union = pred.sum((1,2,3)) + target.sum((1,2,3)) - inter + eps
    return 1 - (inter / union).mean()

bce_loss_fn = nn.BCEWithLogitsLoss()


train_dataset = SegDataSet(
    base_dir=r"xxx/train",
    image_dir="A",
    mask_dir="B"
)

val_dataset = SegDataSet(
    base_dir=r"xxx/val",
    image_dir="A",
    mask_dir="B"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# --------------------------
# EVALUATION FUNCTION
# --------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_bce, total_dice, total_iou = 0, 0, 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)

        total_bce += bce_loss_fn(preds, masks).item()
        total_dice += dice_loss(preds, masks).item()
        total_iou += iou_loss(preds, masks).item()

    n = len(loader)
    return total_bce/n, total_dice/n, total_iou/n

if __name__ == "__main__":
    EPOCHS = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            loss_bce = bce_loss_fn(preds, masks)
            loss_dice = dice_loss(preds, masks)
            loss = loss_bce + loss_dice   # combined loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f}")


        # Run evaluation after each epoch
        bce, dice, iou = evaluate(model, val_loader, device)
        print(f"Eval BCE: {bce:.4f} | DiceLoss: {dice:.4f} | IoULoss: {iou:.4f}")
