"""Train the learned correction of the ball model from resid_data.py samples.

Input: the model's features for one simulated step (geo_controller.js residFeats); target:
how far the real ball's move differed from the model's (x, y, z). The controller adds the
prediction to every simulated step (ctlParams.resid).

    python bot/train_resid.py data/resid_*.jsonl --val-seeds 431-440 --out bot/resid.json
"""
import argparse
import glob
import json

import numpy as np
import torch


def load(paths):
    rows = []
    for p in paths:
        for path in glob.glob(p):
            with open(path) as f:
                rows += [json.loads(line) for line in f]
    return rows


def matrix(rows):
    X = np.array([r["f"] for r in rows], np.float32)
    Y = np.array([r["err"] for r in rows], np.float32)
    ok = np.all(np.abs(Y) < 2, 1) & np.all(np.isfinite(X), 1)   # (bigger: the ball died or respawned)
    return X[ok], Y[ok], [r for r, k in zip(rows, ok) if k]


PHASES = {"rolling": lambda X: (X[:, 0] == 0) & (X[:, 2] == 0), "take-off": lambda X: X[:, 3] == 1,
          "landing": lambda X: X[:, 4] == 1, "flying": lambda X: (X[:, 0] == 1) & (X[:, 2] == 1)}


class Net(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.l1 = torch.nn.Linear(d, h)
        self.l2 = torch.nn.Linear(h, 3)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--val-seeds", default="431-440")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(a.data)
    lo, _, hi = a.val_seeds.partition("-")
    val = set(range(int(lo), int(hi or lo) + 1))
    Xtr, Ytr, _ = matrix([r for r in rows if r["seed"] not in val])
    Xva, Yva, _ = matrix([r for r in rows if r["seed"] in val])
    print(f"{len(Xtr)} training steps, {len(Xva)} validation steps")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    omu, osd = np.zeros(3, np.float32), Ytr.std(0) + 1e-6
    torch.manual_seed(0)
    net = Net(Xtr.shape[1], a.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-5)
    xt, yt = torch.tensor((Xtr - mu) / sd), torch.tensor((Ytr - omu) / osd)
    xv = torch.tensor((Xva - mu) / sd)
    for ep in range(a.epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss = torch.nn.functional.huber_loss(net(xt[b]), yt[b], delta=1.0)
            loss.backward()
            opt.step()
        if ep % 250 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                P = net(xv).numpy() * osd + omu
            res = Yva - P
            print(f"epoch {ep}: validation error (x, y, z) mean |e| {np.abs(Yva).mean(0).round(4)} -> {np.abs(res).mean(0).round(4)}", flush=True)
    with torch.no_grad():
        P = net(xv).numpy() * osd + omu
    for name, sel in PHASES.items():
        m = sel(Xva)
        if m.sum():
            e0, e1 = np.abs(Yva[m]), np.abs(Yva[m] - P[m])
            print(f"  {name:9s} n={m.sum():6d}  mean |error| x {e0[:, 0].mean():.4f}->{e1[:, 0].mean():.4f}  "
                  f"y {e0[:, 1].mean():.4f}->{e1[:, 1].mean():.4f}  z {e0[:, 2].mean():.4f}->{e1[:, 2].mean():.4f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "omu": omu.tolist(), "osd": osd.tolist(),
                       "w1": net.l1.weight.detach().tolist(), "b1": net.l1.bias.detach().tolist(),
                       "w2": net.l2.weight.detach().tolist(), "b2": net.l2.bias.detach().tolist()}, f)
        print("saved", a.out)


if __name__ == "__main__":
    main()
