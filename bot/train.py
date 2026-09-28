"""Train the student CNN on teacher-labelled data (all .npz files under the given dirs)."""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import Policy, choose, flip, stack
from teacher import OBS_H, OBS_W


def targets(vals):
    """Per-action safety targets from planner values.

    vals[a] > 0: number of surviving plans starting with action a (a safety margin);
    vals[a] <= 0: every plan tried after a died. Target 1 = safe, 0 = deadly.
    """
    v = np.asarray(vals, np.float32)
    return np.where(v >= 2, 1.0, np.where(v > 0, 0.85, 0.0)).astype(np.float32)


def load(dirs):
    X, T, W = [], [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            obs, vals = z["obs"], z["vals"]
            if len(obs) < 2 or obs.shape[1:3] != (OBS_H, OBS_W):
                continue
            # Second input frame: the observation one decision step (50ms) earlier.
            prev = z["prev"] if "prev" in z.files else np.concatenate([obs[:1], obs[:-1]])
            X.append(np.stack([stack(o, p) for o, p in zip(obs, prev)]))
            T.append(targets(vals))
            # When nothing is safe the labels say little; otherwise all samples count.
            W.append(np.where(vals.max(1) > 0, 1.0, 0.3).astype(np.float32))
    return np.concatenate(X), np.concatenate(T), np.concatenate(W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="models/student.pt")
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    X, T, W = load(a.dirs)
    print(f"{len(X)} samples; unsafe rate per action {1 - (T > 0.5).mean(0)}", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    nval = max(1, len(X) // 10)
    va, tr = idx[:nval], idx[nval:]
    net = Policy()
    if a.init:
        net.load_state_dict(torch.load(a.init))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    Xt, Tt, Wt = torch.from_numpy(X), torch.from_numpy(T), torch.from_numpy(W)
    swap = torch.tensor([0, 2, 1])
    for ep in range(a.epochs):
        net.train()
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(tr))
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            x, t, w = Xt[b], Tt[b], Wt[b]
            m = torch.rand(len(b)) < 0.5  # mirror half the batch, swapping left/right
            x = torch.where(m[:, None, None, None], flip(x), x)
            t = torch.where(m[:, None], t[:, swap], t)
            bce = F.binary_cross_entropy_with_logits(net(x), t, reduction="none")
            # "Safe" labels are proven by a surviving rollout; "unsafe" only means the limited
            # search found none, so count them a bit less.
            bce = bce * torch.where(t < 0.5, 0.7, 1.0)
            loss = (bce.sum(1) * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            lv = torch.cat([net(Xt[va[i:i + 1024]]) for i in range(0, len(va), 1024)])
            tv = Tt[va] > 0.5
            acc = ((lv > 0) == tv).float().mean(0).tolist()
            # How often the policy's chosen action is one the teacher considers safe,
            # among states where some action is safe.
            pick = torch.from_numpy(np.array([choose(r) for r in torch.sigmoid(lv).numpy()]))
            some = tv.any(1)
            safe = tv[some].gather(1, pick[some][:, None]).float().mean().item()
            nsafe_miss = (~tv[:, 0] & some)
            fix = tv[nsafe_miss].gather(1, pick[nsafe_miss][:, None]).float().mean().item() if nsafe_miss.any() else float("nan")
        print(f"epoch {ep}: loss {tot / len(tr):.4f} safe-acc {[round(a, 3) for a in acc]} "
              f"picked-safe {safe:.3f} (when no-key is deadly: {fix:.3f}) ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
