"""Student policy: a small CNN that maps the last two screen frames to an action."""
import numpy as np
import torch
import torch.nn as nn

from teacher import OBS_H, OBS_W

N_ACTIONS = 3  # none, left, right


class Policy(nn.Module):
    """Outputs one logit per action: the log-odds that taking it now is safe."""

    def __init__(self):
        super().__init__()
        def block(i, o, k):
            return [nn.Conv2d(i, o, k, stride=2, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU()]
        self.conv = nn.Sequential(
            *block(4, 32, 5),   # 48x64
            *block(32, 64, 3),  # 24x32
            *block(64, 64, 3),  # 12x16
            *block(64, 64, 3),  # 6x8
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * (OBS_H // 16) * (OBS_W // 16), 256), nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x):
        # x: (B, 4, H, W) uint8 -> logits
        return self.head(self.conv(x.float() / 255.0))


def choose(p, keep_straight=0.15):
    """Pick an action from per-action safety probabilities: go straight (no key) when that
    is about as safe as the best alternative, else take the safest action."""
    best = int(np.argmax(p))
    if p[0] >= 0.5 and p[0] >= p[best] - keep_straight:
        return 0
    return best


def stack(cur, prev):
    """(H, W, 2) current + previous observation -> (4, H, W) uint8 input."""
    return np.concatenate([cur, prev], axis=2).transpose(2, 0, 1)


def flip(x):
    """Mirror an input batch left/right (pair with swapping the left/right action labels)."""
    return x.flip(-1)


class Student:
    def __init__(self, path=None, device="cpu"):
        self.net = Policy().to(device)
        self.device = device
        if path:
            self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval()
        self.prev = None

    def reset(self):
        self.prev = None

    def logits(self, obs):
        prev = self.prev if self.prev is not None else obs
        self.prev = obs
        x = torch.from_numpy(stack(obs, prev)[None]).to(self.device)
        with torch.no_grad():
            # Average with the mirrored view for a symmetric policy.
            l = self.net(x)[0]
            lf = self.net(flip(x))[0][[0, 2, 1]]
        return ((l + lf) / 2).cpu().numpy()

    def act(self, obs):
        return choose(1.0 / (1.0 + np.exp(-self.logits(obs))))
