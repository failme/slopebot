"""Train the learned plan-safety estimate from geo_data.py samples.

Each sample is one candidate plan at one moment of a real game: the features of the model's
prediction for it (run() in geo_controller.js) and whether its keys really kept the ball alive
(played in the real game). A small network learns P(really survives | features); the
controller combines it with its own score (see ctlParams.safety).

    python bot/train_safety.py data/plans_*.jsonl --val-seeds 10-12 --out bot/safety.json
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

FEATS = ["surv", "danger", "keys", "clear", "air30", "walls", "landings", "airTot", "minEdge", "minObs",
         "bankFrac", "maxVx", "firstAir", "maxVn", "ts", "vz", "startAir", "startBank"]
FAMILIES = ["prev", "none", "x", "lane/", "lane>"]


def family(tag):
    if tag == "prev" or tag == "none":
        return tag
    if tag.startswith("x"):
        return "x"
    return "lane>" if ">" in tag else "lane/"


def load(paths):
    rows = []
    for p in paths:
        for path in glob.glob(p):
            with open(path) as f:
                for line in f:
                    rows.append(json.loads(line))
    return rows


def matrix(rows):
    X = np.array([r["feats"] + [1.0 if family(r["tag"]) == fm else 0.0 for fm in FAMILIES] for r in rows], np.float32)
    y = np.array([1.0 if r["real"] == 999 else 0.0 for r in rows], np.float32)
    return X, y


def auc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)


def decision_quality(rows, p, weights=(0, 50, 100, 200, 400)):
    """At each checkpoint: does the plan picked by score + w * log(p) really survive?"""
    groups = defaultdict(list)
    for r, q in zip(rows, p):
        groups[(r["seed"], r["t"])].append((r, q))
    res = {"any": 0, "chosen": 0, "n": len(groups)}
    for w in weights:
        res[w] = 0
    for g in groups.values():
        if any(r["real"] == 999 for r, _ in g):
            res["any"] += 1
        if any(r["chosen"] and r["real"] == 999 for r, _ in g):
            res["chosen"] += 1
        for w in weights:
            best = max(g, key=lambda rq: rq[0]["sc"] + w * math.log(max(1e-4, rq[1])))
            res[w] += best[0]["real"] == 999
    return res


class Net(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.l1 = torch.nn.Linear(d, h) if h else None
        self.l2 = torch.nn.Linear(h if h else d, 1)

    def forward(self, x):
        if self.l1 is not None:
            x = torch.tanh(self.l1(x))
        return self.l2(x).squeeze(-1)


def train(X, y, h, epochs=300, lr=0.01, wd=1e-4):
    torch.manual_seed(0)
    net = Net(X.shape[1], h)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), yt)
        loss.backward()
        opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--val-seeds", default="10-12")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(a.data)
    lo, _, hi = a.val_seeds.partition("-")
    val = set(range(int(lo), int(hi or lo) + 1))
    tr = [r for r in rows if r["seed"] not in val]
    va = [r for r in rows if r["seed"] in val]
    Xtr, ytr = matrix(tr)
    Xva, yva = matrix(va)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    print(f"{len(tr)} training plans ({ytr.mean():.0%} really survive), {len(va)} validation plans ({yva.mean():.0%})")
    print("model says survives (surv 60) -> really survives:", f"{yva[Xva[:, 0] >= 60].mean():.0%}",
          " model says dies ->", f"{yva[Xva[:, 0] < 60].mean():.0%}")
    for h in (0, a.hidden):
        net = train((Xtr - mu) / sd, ytr, h)
        with torch.no_grad():
            pva = torch.sigmoid(net(torch.tensor((Xva - mu) / sd))).numpy()
            ptr = torch.sigmoid(net(torch.tensor((Xtr - mu) / sd))).numpy()
        print(f"hidden {h}: AUC train {auc(ptr, ytr):.3f} val {auc(pva, yva):.3f}; "
              f"model-survival-only AUC val {auc(Xva[:, 0], yva):.3f}")
        q = decision_quality(va, pva)
        print(f"   validation checkpoints {q['n']}: some plan survives {q['any']}, bot's choice survives {q['chosen']}, "
              + ", ".join(f"score+{w}*log p: {q[w]}" for w in (0, 50, 100, 200, 400)))
        if h == a.hidden and a.out:
            with open(a.out, "w") as f:
                json.dump({"feats": FEATS, "families": FAMILIES, "mu": mu.tolist(), "sd": sd.tolist(),
                           "w1": net.l1.weight.tolist() if net.l1 is not None else None,
                           "b1": net.l1.bias.tolist() if net.l1 is not None else None,
                           "w2": net.l2.weight[0].tolist(), "b2": float(net.l2.bias[0])}, f)
            print("saved", a.out)


if __name__ == "__main__":
    main()
