import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from modelfunc import *
import matplotlib.pyplot as plt
import numpy as np
from dataloader import SegDataSet


def dice_loss(pred, target, eps=1e-6):
    vals = []
    for p, t in zip(pred, target):
        p = torch.sigmoid(p).squeeze(0)
        t = (t.squeeze(0) > 0.5).float()      # force binary
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        vals.append((2 * inter + eps) / (union + eps))
    return 1 - torch.mean(torch.stack(vals))


def iou_loss(pred, target, eps=1e-6):
    s = torch.sigmoid(pred)
    vals = []
    for p, t in zip(s, target):
        p = p.squeeze(0)
        t = (t.squeeze(0) > 0.5).float()      # force binary
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        vals.append((inter + eps) / (union + eps))
    return 1 - torch.mean(torch.stack(vals))


bce_loss_fn = nn.BCEWithLogitsLoss()


def evaluate(model, loader, device):
    model.eval()
    dice_vals, iou_vals, bce_vals = [], [], []
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            dice_vals.append(1 - dice_loss(pred, mask).item())
            iou_vals.append(1 - iou_loss(pred, mask).item())
    return float(np.mean(dice_vals)), float(np.mean(iou_vals)), float(np.mean(bce_vals))


def run_experiment(train_dataset, eval_loader, device, name):
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    epochs = 20

    loss_hist, dice_hist, iou_hist = [], [], []

    for ep in range(epochs):
        model.train()
        total = 0.0

        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()

            with autocast():
                pred = model(img)
                loss = bce_loss_fn(pred, mask) + dice_loss(pred, mask)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()

        avg_loss = total / len(train_loader)
        d, i , b = evaluate(model, eval_loader, device)

        loss_hist.append(avg_loss)
        dice_hist.append(d)
        iou_hist.append(i)

        print(f"[{name}] Epoch {ep+1}/{epochs} | Loss {avg_loss:.4f} | Dice {d:.3f} | IoU {i:.3f} | BCE {b:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, label="Loss")
    plt.plot(dice_hist, label="Dice")
    plt.plot(iou_hist, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}_metrics.png")
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    orig = SegDataSet(r"C:\Users\ad\Desktop\AI\heavy\datasets\brain_tumor_dataset",
                      "train_A", "train_B")

    synt = SegDataSet(r"C:\Users\ad\Desktop\AI\heavy\datasets\brain_tumor_dataset_synt",
                      "images", "masks")

    combo = ConcatDataset([orig, synt])

    evalset = SegDataSet("datasets/brainval", "A", "B")
    eval_loader = DataLoader(evalset, batch_size=1, shuffle=False)

    run_experiment(orig, eval_loader, device, "OriginalOnly")
    run_experiment(combo, eval_loader, device, "OriginalPlusSynthetic")
