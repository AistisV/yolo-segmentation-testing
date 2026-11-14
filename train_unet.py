"""Room-interior segmentation (U-Net), trained on random full-resolution crops.
    python train_unet.py --data /vol/segdata --images /vol/data/data/swiss_dwellings --out /vol/ckpt/unet.pt
"""
import argparse, json, os, random, time
import numpy as np, cv2, torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--crop", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--no-pretrained", action="store_true")
a = ap.parse_args()
dev = "cpu" if a.cpu else "cuda"
plans = json.load(open(os.path.join(a.data, "plans.json")))


def load(ps):
    out = []
    for p in ps:
        mp, ip = os.path.join(a.data, "masks", p + ".png"), os.path.join(plans.get("images", {}).get(p, a.images), p + ".png")
        up = os.path.join(a.data, "upscaled", p + ".png")
        if os.path.exists(up): ip = up                                 # small images were upscaled when the masks were made
        if os.path.exists(mp) and os.path.exists(ip):
            im, m = cv2.imread(ip, cv2.IMREAD_GRAYSCALE), cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if im.shape != m.shape: im = cv2.resize(im, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_CUBIC)
            out.append((im, m))
    return out


train, val = load(plans["train"]), load(plans["val"])
print(len(train), "train plans,", len(val), "val plans", flush=True)
C = a.crop


def sample(data, rng, aug=True):
    img, m = data[rng.randrange(len(data))]
    H, W = img.shape
    if H < C or W < C:                                    # small plans (CubiCasa): pad with white
        img = np.pad(img, ((0, max(0, C - H)), (0, max(0, C - W))), constant_values=255); m = np.pad(m, ((0, max(0, C - H)), (0, max(0, C - W))))
        H, W = img.shape
    for _ in range(10):                                   # prefer crops that contain rooms
        y, x = rng.randrange(max(1, H - C)), rng.randrange(max(1, W - C))
        mc = m[y:y + C, x:x + C]
        if mc.mean() > 10 or rng.random() < 0.1: break
    ic = img[y:y + C, x:x + C].astype(np.float32) / 255.0; mc = (m[y:y + C, x:x + C] > 127).astype(np.float32)
    ic = np.pad(ic, ((0, C - ic.shape[0]), (0, C - ic.shape[1])), constant_values=1.0); mc = np.pad(mc, ((0, C - mc.shape[0]), (0, C - mc.shape[1])))
    if aug:
        k = rng.randrange(4); ic, mc = np.rot90(ic, k), np.rot90(mc, k)
        if rng.random() < 0.5: ic, mc = ic[:, ::-1], mc[:, ::-1]
        ic = ic * rng.uniform(0.8, 1.1) + rng.uniform(-0.08, 0.05)               # contrast / brightness (scans)
        if rng.random() < 0.3: ic = cv2.GaussianBlur(np.ascontiguousarray(ic), (3, 3), rng.uniform(0.3, 1.0))
        if rng.random() < 0.3: ic = ic + np.random.normal(0, rng.uniform(0.01, 0.05), ic.shape)
    return np.ascontiguousarray(ic, dtype=np.float32), np.ascontiguousarray(mc, dtype=np.float32)


model = smp.Unet("resnet34", encoder_weights=None if a.no_pretrained else "imagenet", in_channels=1, classes=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
dice = smp.losses.DiceLoss("binary")
rng = random.Random(0); t0 = time.time()
for step in range(1, a.steps + 1):
    xb, yb = zip(*[sample(train, rng) for _ in range(a.bs)])
    x = torch.from_numpy(np.stack(xb))[:, None].to(dev); y = torch.from_numpy(np.stack(yb))[:, None].to(dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=not a.cpu):
        logit = model(x)
    logit = logit.float()
    loss = F.binary_cross_entropy_with_logits(logit, y) + dice(logit, y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 200 == 0 or step == a.steps:
        model.eval(); vr = random.Random(1); inter = union = 0
        with torch.no_grad():
            for _ in range(8):
                xv, yv = zip(*[sample(val or train, vr, aug=False) for _ in range(8)])
                pv = model(torch.from_numpy(np.stack(xv))[:, None].to(dev)).float().sigmoid().cpu().numpy()[:, 0] > 0.5
                yv = np.stack(yv) > 0.5; inter += (pv & yv).sum(); union += (pv | yv).sum()
        model.train()
        print(f"step {step} loss {loss.item():.4f} val IoU {inter / max(union, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), a.out)
print("saved", a.out)
