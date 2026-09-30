# -*- coding: utf-8 -*-
"""生成两个面试演示视频:

1. demo_trajectories.mp4  同一观测 + 5 种语言指令, 轨迹逐帧分化动画
2. demo_first_person.mp4  同一观测, turn left vs turn right 第一视角对比
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imageio

from data import build_scene
from model import MiniVLA
from vla_data import VLADataSet, VOCAB
from visualize import sim_trajectory

torch.manual_seed(0)
np.random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "videos"
os.makedirs(OUT, exist_ok=True)


def fig_to_np(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def load_model():
    ck = torch.load("ckpt/vla_best.pt", map_location="cpu")
    args = ck["args"]
    m = MiniVLA(img_size=args["size"], patch=8, dim=args["dim"],
                depth=args["depth"], Ta=args["Ta"], vocab=len(VOCAB))
    m.load_state_dict(ck.get("ema", ck["model"]))
    m.eval()
    return m, args


model, args = load_model()
boxes = build_scene(0)
ds = VLADataSet(256, args["Ta"], args["size"], seed=777, boxes=boxes)
o, _, _ = ds[0]
START = (2.0, -3.0, 0.7)

# ---------- 视频 1: 轨迹分化动画 ----------
COLORS = {"go straight": "tab:blue", "turn left": "tab:green",
          "turn right": "tab:red", "move forward fast": "tab:orange",
          "move forward slowly": "tab:purple"}
STEPS = 64
print("[video1] sampling trajectories ...")
trajs = {}
for txt in COLORS:
    xs, ys, _, _ = sim_trajectory(model, o, START, txt, boxes,
                                  steps=STEPS, size=o.shape[-1])
    trajs[txt] = (xs, ys)
    print(f"  {txt}: {len(xs)} pts")

print("[video1] rendering frames ...")
w1 = imageio.get_writer(os.path.join(OUT, "demo_trajectories.mp4"),
                        fps=10, codec="libx264", quality=8,
                        macro_block_size=1)
for t in range(2, STEPS + 2):
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    for txt, c in COLORS.items():
        xs, ys = trajs[txt]
        n = min(t, len(xs))
        ax.plot(xs[:n], ys[:n], color=c, lw=2.5, label=txt)
        ax.scatter(xs[n - 1], ys[n - 1], color=c, s=80, zorder=5,
                   edgecolors="k", linewidths=0.8)
    ax.scatter(trajs["go straight"][0][0], trajs["go straight"][1][0],
               c="k", s=70, zorder=6, marker="s")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.grid(alpha=0.3)
    ax.set_title(f"同一观测 + 不同语言指令 → 不同轨迹  (t={t})", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    w1.append_data(fig_to_np(fig))
w1.close()
print("[video1] saved demo_trajectories.mp4")


# ---------- 视频 2: turn left vs turn right 第一视角 ----------
def up(img, k=4):
    arr = img.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    return np.asarray(im.resize((im.width * k, im.height * k), Image.NEAREST))


STEPS2 = 48
print("[video2] sampling left/right runs ...")
_, _, _, fr_l = sim_trajectory(model, o, START, "turn left", boxes,
                               steps=STEPS2, size=o.shape[-1])
_, _, _, fr_r = sim_trajectory(model, o, START, "turn right", boxes,
                               steps=STEPS2, size=o.shape[-1])
n = min(len(fr_l), len(fr_r))

print("[video2] rendering frames ...")
w2 = imageio.get_writer(os.path.join(OUT, "demo_first_person.mp4"),
                        fps=8, codec="libx264", quality=8,
                        macro_block_size=1)
for i in range(n):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=100)
    axes[0].imshow(up(fr_l[i]))
    axes[0].axis("off")
    axes[0].set_title("指令: turn left", fontsize=12)
    axes[1].imshow(up(fr_r[i]))
    axes[1].axis("off")
    axes[1].set_title("指令: turn right", fontsize=12)
    fig.suptitle(f"同一观测 · 相反指令 → 第一视角反向旋转  (t={i})",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    w2.append_data(fig_to_np(fig))
w2.close()

for f in ("demo_trajectories.mp4", "demo_first_person.mp4"):
    p = os.path.join(OUT, f)
    print(f"[done] {p}  {os.path.getsize(p)/1e6:.2f} MB")
