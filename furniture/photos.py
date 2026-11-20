"""Photo inputs for the ABO listings: image loading, pretrained ResNet-18 features, and a small CNN.

All photos are ABO catalogue images (Amazon.com, CC BY 4.0), 256 px, in data/external/abo/images/small/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "external" / "abo" / "images" / "small"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet statistics
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def photo_views(listings, generic):
    """One row per (listing, photo): main photo first, then up to 3 other views; generic photos dropped.
    `row` is the listing's position in `listings`."""
    man = pd.read_csv(ROOT / "data" / "processed" / "abo_images.csv")
    man = man[man["ok"] & ~man["image_id"].isin(generic)]
    key = listings[["item_id", "domain"]].reset_index(drop=True).reset_index().rename(columns={"index": "row"})
    return man.merge(key, on=["item_id", "domain"]).sort_values(["row", "view"]).reset_index(drop=True)


def load_images(paths, size):
    """Paths -> uint8 tensor (N, 3, size, size). The photo is padded to a square on white, then resized,
    so the furniture's proportions are kept (they carry the same information as the sizes)."""
    out = torch.empty((len(paths), 3, size, size), dtype=torch.uint8)
    for i, p in enumerate(paths):
        im = Image.open(IMG_DIR / p).convert("RGB")
        side = max(im.size)
        sq = Image.new("RGB", (side, side), (255, 255, 255))
        sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        out[i] = torch.from_numpy(np.array(sq.resize((size, size), Image.BILINEAR))).permute(2, 0, 1)
    return out


def to_float(x):
    return ((x.float() / 255) - MEAN) / STD


@torch.no_grad()
def resnet18_features(images, batch=64):
    """512-number description of each photo from ResNet-18 pretrained on ImageNet, last layer removed.
    The network is frozen: nothing here is fitted on our data, so it cannot leak between folds."""
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    feats = [net(to_float(images[i:i + batch])) for i in range(0, len(images), batch)]
    return torch.cat(feats).numpy()


class SmallCNN(nn.Module):
    """Four convolution blocks (conv - batch norm - ReLU - max pool), then global average pooling and one
    linear layer. About 0.1 M weights, trained from scratch."""

    def __init__(self, n_classes, width=16):
        super().__init__()
        chans = [3, width, width * 2, width * 4, width * 8]
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                          nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True), nn.MaxPool2d(2))
            for i in range(4)])
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(chans[-1], n_classes)

    def forward(self, x):
        x = self.blocks(x)
        return self.head(self.drop(x.mean(dim=(2, 3))))


def augment(x):
    """Random horizontal flip and a small random shift (pad 4, crop back), applied per batch."""
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    pad = F.pad(x, (4, 4, 4, 4), mode="replicate")
    i, j = np.random.randint(0, 9, size=2)
    return pad[:, :, i:i + x.shape[2], j:j + x.shape[3]]


def train_small_cnn(x_train, y_train, n_classes, epochs=12, batch=64, lr=3e-3, seed=42, x_val=None, y_val=None,
                    log=None):
    """Train SmallCNN on uint8 images and integer labels. Class weights = inverse class frequency, so the
    small types count as much as the large ones (the same goal as macro-F1). Returns the model and, if
    validation data is given, the per-epoch history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCNN(n_classes)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(x_train) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    yt = torch.as_tensor(y_train)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            xb = augment(to_float(x_train[idx]))
            loss = F.cross_entropy(model(xb), yt[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "train loss": total / len(x_train)}
        if x_val is not None:
            p = predict_proba(model, x_val)
            row["validation loss"] = float(F.nll_loss(torch.log(torch.as_tensor(p) + 1e-9), torch.as_tensor(y_val)))
            row["validation accuracy"] = float((p.argmax(1) == y_val).mean())
        history.append(row)
        if log:
            log(row)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_proba(model, x, batch=256):
    model.eval()
    return torch.cat([F.softmax(model(to_float(x[i:i + batch])), 1) for i in range(0, len(x), batch)]).numpy()


def cached_resnet18_features(image_ids, paths, cache=ROOT / "data" / "cache" / "abo_resnet18.npz"):
    """ResNet-18 features per unique photo, computed once and stored in data/cache (delete it to recompute)."""
    ids = pd.Series(paths.values, index=image_ids.values)
    ids = ids[~ids.index.duplicated()]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        known = dict(zip(z["ids"], z["feats"]))
        if all(i in known for i in ids.index):
            return np.stack([known[i] for i in image_ids])
    feats = []
    for k in range(0, len(ids), 512):
        feats.append(resnet18_features(load_images(ids.values[k:k + 512].tolist(), 224)))
    feats = np.concatenate(feats)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.array(ids.index), feats=feats)
    known = dict(zip(ids.index, feats))
    return np.stack([known[i] for i in image_ids])
