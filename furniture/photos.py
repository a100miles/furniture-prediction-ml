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


def load_images(paths, size, img_dir=None):
    """Paths -> uint8 tensor (N, 3, size, size). The photo is padded to a square on white, then resized,
    so the furniture's proportions are kept (they carry the same information as the sizes)."""
    out = torch.empty((len(paths), 3, size, size), dtype=torch.uint8)
    for i, p in enumerate(paths):
        im = Image.open((img_dir or IMG_DIR) / p).convert("RGB")
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


class FineTunedResNet18(nn.Module):
    """ResNet-18 from ImageNet with a new 9-type output layer. The stem and the first two blocks stay frozen
    (generic edges and textures); blocks 3-4 and the output layer are trained on our photos."""

    def __init__(self, n_classes):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))
        self.frozen = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
        self.tuned = nn.Sequential(net.layer3, net.layer4)
        self.head = net.fc
        for p in self.frozen.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()                     # frozen batch-norm keeps the ImageNet statistics
        return self

    def forward(self, x):
        with torch.no_grad():
            x = self.frozen(x)
        x = self.tuned(x)
        return self.head(x.mean(dim=(2, 3)))


def train_finetune(x_train, y_train, n_classes, epochs=4, batch=48, lr=1e-3, seed=42, log=None):
    """Fine-tune FineTunedResNet18 on uint8 images (any square size; 160 px used). Same class weights,
    augmentation and one-cycle schedule as the small CNN; the pretrained blocks get a 10x smaller learning rate."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FineTunedResNet18(n_classes)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    opt = torch.optim.AdamW([{"params": model.tuned.parameters(), "lr": lr / 10},
                             {"params": model.head.parameters(), "lr": lr}], weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(x_train) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr / 10, lr], total_steps=steps)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            loss = F.cross_entropy(model(augment(to_float(x_train[idx]))), yt[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if log:
            log({"epoch": epoch, "train loss": total / len(x_train)})
    return model


@torch.no_grad()
def predict_proba(model, x, batch=256):
    model.eval()
    return torch.cat([F.softmax(model(to_float(x[i:i + batch])), 1) for i in range(0, len(x), batch)]).numpy()


# Stronger frozen encoders (Final, after the survey): both are pretrained on far more images than ImageNet's
# 1.3 M and are used without any fitting on our data, like ResNet-18.
#   CLIP ViT-B/32 (OpenAI, MIT licence): image tower trained to match 400 M images with their captions.
#   DINOv2 ViT-S/14 (Meta, Apache 2.0): self-supervised on 142 M curated images, no labels or captions.
ENCODERS = {
    "clip": ("openai/clip-vit-base-patch32", (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
    "dinov2": ("facebook/dinov2-small", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
_NETS = {}


def _encoder(name):
    if name not in _NETS:
        hub_id = ENCODERS[name][0]
        if name == "clip":
            from transformers import CLIPVisionModelWithProjection
            net = CLIPVisionModelWithProjection.from_pretrained(hub_id)
        else:
            from transformers import AutoModel
            net = AutoModel.from_pretrained(hub_id)
        _NETS[name] = net.eval()
    return _NETS[name]


@torch.no_grad()
def encoder_features(name, images, batch=64):
    """Frozen embedding per photo: CLIP image embedding (512) or DINOv2 class token (384).
    `images` are uint8 (N, 3, 224, 224) from load_images (white square padding, as for ResNet-18)."""
    if name == "resnet18":
        return resnet18_features(images, batch)
    net = _encoder(name)
    mean = torch.tensor(ENCODERS[name][1]).view(1, 3, 1, 1)
    std = torch.tensor(ENCODERS[name][2]).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = ((images[i:i + batch].float() / 255) - mean) / std
        if name == "clip":
            out.append(net(pixel_values=x).image_embeds)
        else:
            out.append(net(pixel_values=x).last_hidden_state[:, 0])
    return torch.cat(out).numpy()


def cached_features(encoder, image_ids, paths, cache=None, img_dir=None):
    """Features of one encoder per unique photo, computed once and stored in data/cache (delete to recompute)."""
    cache = cache or ROOT / "data" / "cache" / f"abo_{encoder}.npz"
    if encoder == "resnet18" and img_dir is None:
        return cached_resnet18_features(image_ids, paths, cache)
    ids = pd.Series(paths.values, index=image_ids.values)
    ids = ids[~ids.index.duplicated()]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        known = dict(zip(z["ids"], z["feats"]))
        if all(i in known for i in ids.index):
            return np.stack([known[i] for i in image_ids])
    feats = []
    for k in range(0, len(ids), 256):
        feats.append(encoder_features(encoder, load_images(ids.values[k:k + 256].tolist(), 224, img_dir)))
    feats = np.concatenate(feats)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.array(ids.index), feats=feats)
    known = dict(zip(ids.index, feats))
    return np.stack([known[i] for i in image_ids])


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
