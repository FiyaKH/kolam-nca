"""Neural Cellular Automata that grows (and regrows) a kolam, designed to train on a laptop CPU.
Based on Mordvintsev et al., "Growing Neural Cellular Automata", Distill 2020.

Training can be split over several sessions: it saves a checkpoint every 25 iterations and when you
press Ctrl+C, and the next run continues from there automatically.

  python nca_kolam.py --preview          save the target kolam (out/target.png) and exit
  python nca_kolam.py                    train (continues from out/checkpoint.pt if it exists)
  python nca_kolam.py --demo             only make the grow-and-regrow GIF from the latest checkpoint
  python nca_kolam.py --fresh            start training again from zero
  python nca_kolam.py --iters 3500 --lr 0.0005
                                         continue an earlier run for more iterations at a chosen learning rate
"""
import argparse, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=1500, help="total training iterations (across all sessions)")
p.add_argument("--size", type=int, default=48)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--device", default="cpu", choices=["cpu", "xpu", "cuda"])
p.add_argument("--preview", action="store_true", help="only save the target image and exit")
p.add_argument("--demo", action="store_true", help="only make the GIF from the latest checkpoint")
p.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start from zero")
p.add_argument("--lr", type=float, default=None, help="override the learning rate when continuing a run")
args = p.parse_args()
torch.manual_seed(0); np.random.seed(0)
device = args.device
SIZE, PAD, CH = args.size, 8, 16
H = W = SIZE + 2 * PAD
CKPT = "out/checkpoint.pt"
os.makedirs("out", exist_ok=True)

# ---------- 1. Target: an original kolam with bold lines (readable at 48 px) ----------
def make_bold_kolam(size, scale=12, color=(245, 240, 230, 255)):
    """Original kolam with bold lines, designed to stay readable at ~48 px."""
    S = size * scale; c = S / 2
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    W = lambda w: max(1, int(S * w))
    P = lambda r, a: (c + S * r * np.cos(a), c + S * r * np.sin(a))
    def line(pts, w):
        pts = list(pts); d.line(pts + [pts[0]], fill=color, width=W(w), joint="curve")
    def dot(r, a, rad):
        x, y = P(r, a); rr = S * rad; d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=color)
    def petal(r0, r1, a, hw, w, shrink=0.0):
        s = np.linspace(0, 1, 80)
        prof = np.sin(np.pi * s) ** 0.6 * (1 - 0.45 * s)
        rr = r0 + (r1 - r0) * (shrink + (1 - 2 * shrink) * s)
        half = hw * (1 - shrink) * prof
        line([P(r, a - h) for r, h in zip(rr, half)] + [P(r, a + h) for r, h in zip(rr[::-1], half[::-1])], w)
    t = np.linspace(0, 2 * np.pi, 3000)
    dot(0, 0, 0.03)                                             # centre dot
    for k in range(8):                                          # 8 centre petals
        petal(0.055, 0.165, k * np.pi / 4, 0.28, 0.026)
    line([P(0.205, a) for a in t], 0.026)                        # ring
    for k in range(8):                                          # 8 big outer petals, each with a dot inside
        a = k * np.pi / 4 + np.pi / 8
        petal(0.235, 0.47, a, 0.24, 0.03)
        dot(0.345, a, 0.022)
    for k in range(8):                                          # dots between the big petals
        dot(0.33, k * np.pi / 4, 0.02)
        dot(0.43, k * np.pi / 4, 0.016)
    return img

def make_target(size):
    img = make_bold_kolam(size).resize((size, size), Image.LANCZOS)
    a = np.asarray(img).astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]                                 # premultiplied alpha
    t = torch.tensor(a).permute(2, 0, 1)                      # 4 x size x size
    return F.pad(t, (PAD, PAD, PAD, PAD))

target = make_target(SIZE).to(device)
FLOOR = torch.tensor([92, 62, 45], dtype=torch.float32).view(3, 1, 1) / 255   # dark floor colour

def to_rgb(x):                                                # state -> RGB image on the floor
    rgb, a = x[:3].clamp(0, 1).cpu(), x[3:4].clamp(0, 1).cpu()
    img = ((1 - a) * FLOOR + rgb).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8)).resize((W * 6, H * 6), Image.NEAREST)

to_rgb(target).save("out/target.png")
if args.preview:
    print("Saved out/target.png (preview only)"); raise SystemExit

# ---------- 2. The NCA: every cell looks only at its 3x3 neighbourhood ----------
class NCA(nn.Module):
    def __init__(self, ch=CH, hidden=96, fire_rate=0.5):
        super().__init__()
        self.ch, self.fire_rate = ch, fire_rate
        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        ident = torch.zeros(3, 3); ident[1, 1] = 1
        k = torch.stack([ident, sobel, sobel.T])              # identity, d/dx, d/dy
        self.register_buffer("kernels", k.repeat(ch, 1, 1).unsqueeze(1))
        self.fc1 = nn.Conv2d(ch * 3, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)   # start as "do nothing"

    def alive(self, x):                                       # a cell is alive if it or a neighbour is visible
        return F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.1

    def forward(self, x):
        pre = self.alive(x)
        y = F.conv2d(x, self.kernels, padding=1, groups=self.ch)        # look: own value + Sobel x, y
        dx = self.fc2(F.relu(self.fc1(y)))                                # think: how to change 16 values
        fire = (torch.rand(x[:, :1].shape, device=x.device) <= self.fire_rate).float()
        x = x + dx * fire                                                 # coin flip: half the cells update
        return x * (pre & self.alive(x)).float()                          # alive check

def seed_state(n=1):
    s = torch.zeros(n, CH, H, W, device=device)
    s[:, 3:, H // 2, W // 2] = 1.0                            # one living, colourless cell in the centre
    return s

def damage(x):                                                # erase a small bite, or a whole half
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    for i in range(x.shape[0]):
        if np.random.rand() < 0.5:                            # small bite: a random circle
            cy, cx = np.random.randint(PAD, H - PAD, 2); r = np.random.randint(SIZE // 6, SIZE // 3)
            x[i, :, ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r] = 0
        else:                                                 # big damage: cut away the left, right, top or bottom half
            side = np.random.randint(4)
            if side == 0:   x[i, :, :, W // 2:] = 0
            elif side == 1: x[i, :, :, :W // 2] = 0
            elif side == 2: x[i, :, H // 2:, :] = 0
            else:           x[i, :, :H // 2, :] = 0
    return x

model = NCA().to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(args.iters * 0.7)], 0.1)
pool = seed_state(256)
start_it = 0

# ---------- 3. Resume from the last session, if there is one ----------
if os.path.exists(CKPT) and not args.fresh:
    ck = torch.load(CKPT, map_location=device, weights_only=False)   # our own file, safe to load fully
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    pool = ck["pool"].to(device); start_it = ck["iter"]
    print(f"Resumed from iteration {start_it} (latest loss {ck.get('loss', float('nan')):.5f})")

if args.lr is not None:                                       # e.g. a higher learning rate to learn a new skill
    for g in opt.param_groups:
        g["lr"] = args.lr
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.7 * (args.iters - start_it))], 0.1)   # counts from this run's start
    print(f"Learning rate set to {args.lr} (dropped to {args.lr * 0.1:g} for the last 30% of this run)")

def save_checkpoint(it, loss_value):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "pool": pool.cpu(), "iter": it, "loss": loss_value}, CKPT)

# ---------- 4. Training (pool + seeds + damage) ----------
if not args.demo and start_it < args.iters:
    n_damage = max(1, args.batch // 2)                        # half the batch practises repair
    lo, hi = int(1.4 * SIZE), int(1.8 * SIZE)                 # steps per attempt (enough to grow to full size)
    print(f"Device: {device} | grid {H}x{W} | batch {args.batch} | iterations {start_it} -> {args.iters} "
          f"| {lo}-{hi} steps per attempt | Ctrl+C saves and stops")
    t0, it, loss_value = time.time(), start_it, float("nan")
    try:
        for it in range(start_it + 1, args.iters + 1):
            idx = torch.randint(0, len(pool), (args.batch,), device=device)
            x = pool[idx].clone()
            with torch.no_grad():
                order = ((x[:, :4] - target) ** 2).mean((1, 2, 3)).argsort(descending=True)
                x = x[order]
                x[:1] = seed_state(1)                         # worst sample -> fresh seed (keep practising growth)
                if it > args.iters // 4:
                    x[-n_damage:] = damage(x[-n_damage:])     # damage the best samples (practise repair)
            for _ in range(np.random.randint(lo, hi + 1)):
                x = model(x)
            loss = ((x[:, :4] - target) ** 2).mean()
            opt.zero_grad(); loss.backward()
            for prm in model.parameters():                    # gradient normalisation (stabilises training)
                prm.grad = prm.grad / (prm.grad.norm() + 1e-8)
            opt.step(); sched.step()
            pool[idx[order]] = x.detach()
            loss_value = loss.item()
            if it % 25 == 0:
                save_checkpoint(it, loss_value)
                done = it - start_it; el = time.time() - t0
                print(f"iter {it:5d} | loss {loss_value:.5f} | {el / done:.2f} s/iter | "
                      f"~{el / done * (args.iters - it) / 60:.0f} min to finish | saved")
    except KeyboardInterrupt:
        print("Stopping: saving checkpoint...")
    save_checkpoint(it, loss_value)
    print(f"Checkpoint saved at iteration {it}. Run the same command again to continue.")

# ---------- 5. Demo: grow from one cell, then cut it in half and watch it regrow ----------
model.eval(); frames = []
with torch.no_grad():
    x = seed_state(1)
    G = 3 * SIZE                                              # steps to grow, then the same to regrow
    for step in range(2 * G):
        if step == G:
            grown = to_rgb(x[0])
            x[:, :, :, W // 2:] = 0                           # cut away the right half
            damaged = to_rgb(x[0])
        x = model(x)
        if step % 2 == 0:
            frames.append(to_rgb(x[0]))
frames[0].save("out/grow_and_regrow.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
grown.save("out/grown.png"); damaged.save("out/after_damage.png"); frames[-1].save("out/regrown.png")
print("Saved out/grow_and_regrow.gif, out/grown.png, out/after_damage.png, out/regrown.png")