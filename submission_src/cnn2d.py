"""Shared 2.5D input builder + model. Imported by training and by main.py at inference."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# three axial slabs over the striatal complex (z centre = 24 in the 48-slice crop)
SLABS = ((18, 22), (22, 26), (26, 30))
SCALE = 4.0


def make_input(crop: np.ndarray, zshift: int = 0) -> np.ndarray:
    """crop: (2, 48, 64, 64) -> (3, 64, 64) float32. Uses channel 1 (background-normalised)."""
    ch = np.asarray(crop[1], dtype=np.float32)
    out = np.empty((3, ch.shape[1], ch.shape[2]), dtype=np.float32)
    for i, (z0, z1) in enumerate(SLABS):
        out[i] = ch[z0 + zshift:z1 + zshift].mean(axis=0)
    return out / SCALE - 0.5


class Net(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=weights)
        self.resnet.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.interpolate(x, size=128, mode="bilinear", align_corners=False)
        return self.resnet(x)


def build_pretrained() -> "Net":
    return Net(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)


def load_model(path, device="cpu") -> "Net":
    m = Net(weights=None)
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    return m.to(device).eval()


class NetT(nn.Module):
    """resnet18 truncated after layer3 (256 ch) — ~1/8 the weights of Net."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = F.interpolate(x, size=128, mode="bilinear", align_corners=False)
        return self.fc(self.pool(self.stem(x)).flatten(1))


def _load_dir(d, ctor, device):
    """Load every *.pt in d into a fresh ctor(); dequantize_sd handles int8 and fp16 alike."""
    from pathlib import Path
    from quant import dequantize_sd
    models = []
    for p in sorted(Path(d).glob("*.pt")):
        m = ctor()
        m.load_state_dict(dequantize_sd(torch.load(p, map_location="cpu")))
        models.append(m.to(device).eval())
    return models


@torch.no_grad()
def _mean_tta(models, x, device, batch=64):
    out = np.zeros(len(x), np.float64)
    for m in models:
        chunks = []
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            lg = m(xb).squeeze(1) + m(torch.flip(xb, dims=[-1])).squeeze(1)
            chunks.append((lg / 2.0).float().cpu().numpy())
        out += np.concatenate(chunks)
    return out / len(models)


def make_input6(crop, zshift: int = 0):
    """(2,48,64,64) -> (6,64,64): ch0-2 = background-normalised slabs, ch3-5 = peak-normalised slabs."""
    a = make_input(crop, zshift)
    ch = np.asarray(crop[0], dtype=np.float32)
    b = np.empty_like(a)
    for i, (z0, z1) in enumerate(SLABS):
        b[i] = ch[z0 + zshift:z1 + zshift].mean(axis=0)
    return np.concatenate([a, b - 0.5], axis=0)


class NetT6(NetT):
    """NetT with a 6-channel stem; conv1 starts as the mean over the two channel groups."""

    def __init__(self, pretrained: bool = False):
        super().__init__(pretrained=pretrained)
        w = self.stem[0].weight.data
        conv = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        conv.weight.data = torch.cat([w, w], dim=1) / 2.0
        self.stem[0] = conv


_SPECS = (
    ("cnn_2dt", "2dt", lambda: NetT(pretrained=False), 3),
    ("cnn_2d", "2d", lambda: Net(weights=None), 3),
    ("cnn_2dtc", "2dtc", lambda: NetT(pretrained=False), 3),
    ("cnn_2dt6", "2dt6", lambda: NetT6(pretrained=False), 6),
    ("cnn_2dtc5", "2dtc5", lambda: NetT(pretrained=False), 3),
    ("cnn_2dt6n", "2dt6n", lambda: NetT6(pretrained=False), 6),
    ("cnn_2dt6c5", "2dt6c5", lambda: NetT6(pretrained=False), 6),
)


def predict_axial(crops, device):
    """crops: list of valid (2,48,64,64) arrays -> {key: raw mean-TTA logits} for each model dir present."""
    from pathlib import Path
    src = Path(__file__).resolve().parent
    xs = {}
    out = {}
    for key, d, ctor, nch in _SPECS:
        mdir = src / "models" / d
        if not any(mdir.glob("*.pt")):
            continue
        if nch not in xs:
            f = make_input if nch == 3 else make_input6
            xs[nch] = np.stack([f(c) for c in crops]).astype(np.float32)
        out[key] = _mean_tta(_load_dir(mdir, ctor, device), xs[nch], device)
    return out
