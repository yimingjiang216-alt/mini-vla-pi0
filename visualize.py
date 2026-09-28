# -*- coding: utf-8 -*-
"""迷你 VLA 可视化: 语言指令驱动的轨迹生成 + 与世界模型的闭环.

出三张图:
  1. 指令->轨迹: 同一起点, 不同指令下 VLA 生成的轨迹(俯视图),
     直观展示"同一观测, 换指令 -> 换行为".
  2. 动作对比条形图: 各指令预测角速度 vs 真值.
  3. 世界模型闭环: 用 VLA 生成的动作驱动世界模型场景渲染器,
     生成"假想未来帧", 展示 VLA->世界模型的数据流向.
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_scene, render_frame
from model import MiniVLA
from vla_data import VLADataSet, encode_instruction, VOCAB

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def enc_txt(text):
    return torch.from_numpy(encode_instruction(text))


def sim_trajectory(model, start_obs, start_state, lang_text, boxes,
                   steps=24, Ta=8, size=48):
    """用 VLA 生成的动作块驱动仿真环境, 得到轨迹与渲染帧.

    动作 -> 状态转移(与世界模型项目同一动力学):
        x' = x + v cos(th); y' = y + v sin(th); th' = th + omega
    """
    model.eval()
    x, y, th = start_state
    xs, ys, ths = [x], [y], [th]
    frames = [start_obs]
    lang = enc_txt(lang_text).unsqueeze(0)
    obs = start_obs.unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps // Ta + 1):
            act = model.sample(obs, lang, steps=10)   # (1, Ta, 2)
            for k in range(act.size(1)):
                v = float(act[0, k, 0]) * 3.0
                om = float(act[0, k, 1]) * 0.9
                x = np.clip(x + v * np.cos(th), -38, 38)
                y = np.clip(y + v * np.sin(th), -38, 38)
                th = th + om
                img, _ = render_frame((x, y, th), boxes, size, size)
                xs.append(x); ys.append(y); ths.append(th)
                frames.append(img)
                obs = torch.from_numpy(img).unsqueeze(0)
    return np.array(xs), np.array(ys), np.array(ths), np.stack(frames)


def fig_trajectories(model, ds, boxes, out, n_start=4, steps=32):
    fig, axes = plt.subplots(1, n_start, figsize=(5 * n_start, 5))
    if n_start == 1:
        axes = [axes]
    for si in range(n_start):
        ax = axes[si]
        o, l, a = ds[si * 7]
        # 用真值动作反推近似起点状态: 从轨迹平均速度估
        v0 = float(a[:, 0].mean()) * 3.0
        th0 = 0.0
        # 观测的起点状态由数据集内部决定, 这里以场景中心附近为起点
        start = (float(np.random.default_rng(si).uniform(-10, 10)),
                 float(np.random.default_rng(si + 9).uniform(-10, 10)),
                 float(np.random.default_rng(si + 3).uniform(0, 2 * np.pi)))
        ax.set_title(f"起点 {si+1}", fontsize=12)
        colors = {"go straight": "tab:blue", "turn left": "tab:green",
                  "turn right": "tab:red", "move forward fast": "tab:orange",
                  "move forward slowly": "tab:purple"}
        for txt, c in colors.items():
            xs, ys, _, _ = sim_trajectory(model, o, start, txt, boxes,
                                          steps=steps, size=o.shape[-1])
            ax.plot(xs, ys, color=c, lw=2.5, alpha=0.9, label=txt)
            ax.scatter(xs[0], ys[0], c="k", s=40, zorder=5)
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30)
        ax.grid(alpha=0.3)
        if si == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("迷你 VLA: 同一起点, 不同语言指令 -> 不同轨迹", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trajectories.png"), dpi=130)
    plt.close(fig)


def fig_action_bars(model, ds, out, n=96):
    texts = ["go straight", "turn left", "turn right",
             "move forward fast", "move forward slowly"]
    preds = {t: [] for t in texts}
    gts = {t: [] for t in texts}
    idx = np.random.default_rng(5).choice(len(ds), size=min(n, len(ds)),
                                          replace=False)
    with torch.no_grad():
        for i in idx:
            o, l, a = ds[int(i)]
            obs = o.unsqueeze(0)
            for t in texts:
                p = model.sample(obs, enc_txt(t).unsqueeze(0), steps=10)
                preds[t].append(p[0].numpy())
                # 每条样本的真值动作只取一次(真值不随指令改变, 用于基线对比)
                if len(gts[t]) <= 0:
                    gts[t].append(a.numpy())
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.35
    for k, t in enumerate(texts):
        P = np.stack(preds[t]); G = np.stack(gts[t])
        pv, gv = P[..., 0].mean(), G[..., 0].mean()
        po, go = P[..., 1].mean(), G[..., 1].mean()
        ax.bar(k - w / 2, gv, w, color="tab:gray", label="真值 v" if k == 0 else "")
        ax.bar(k + w / 2, pv, w, color="tab:cyan", label="预测 v" if k == 0 else "")
        ax.bar(k - w / 2 + 5, go, w, color="tab:olive", label="真值 ω" if k == 0 else "")
        ax.bar(k + w / 2 + 5, po, w, color="tab:pink", label="预测 ω" if k == 0 else "")
    ax.set_xticks(range(len(texts) + 1))
    ax.set_xticklabels(texts + [""], rotation=15, fontsize=9)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("归一化动作值")
    ax.set_title("各指令下预测动作 vs 真值 (左:线速度 v / 右:角速度 ω)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "action_bars.png"), dpi=130)
    plt.close(fig)
    return {t: {"pred_omega": float(np.stack(preds[t])[..., 1].mean()),
                "gt_omega": float(np.stack(gts[t])[..., 1].mean())}
            for t in texts}


def fig_world_model_loop(model, ds, boxes, out, lang="turn left", nrow=6):
    """VLA 动作驱动场景渲染 -> 世界模型视角的"想象未来"."""
    o, l, a = ds[3]
    size = o.shape[-1]
    start = (5.0, -5.0, 0.6)
    xs, ys, ths, frames = sim_trajectory(model, o, start, lang, boxes,
                                         steps=24, size=size)
    fig, axes = plt.subplots(1, nrow, figsize=(2.6 * nrow, 3))
    sel = np.linspace(0, len(frames) - 1, nrow).astype(int)
    for k, s in enumerate(sel):
        axes[k].imshow(frames[s].transpose(1, 2, 0))
        axes[k].axis("off")
        axes[k].set_title(f"t={s}", fontsize=10)
    fig.suptitle(f"世界模型闭环: VLA 在指令 \"{lang}\" 下生成的想象未来帧",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "world_model_loop.png"), dpi=130)
    plt.close(fig)
    return len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt/best.pt")
    p.add_argument("--out", type=str, default="figs")
    p.add_argument("--n_start", type=int, default=4)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    args = ck["args"]
    model = MiniVLA(img_size=args["size"], patch=8, dim=args["dim"],
                    depth=args["depth"], Ta=args["Ta"], vocab=len(VOCAB))
    model.load_state_dict(ck.get("ema", ck["model"]))
    model.eval()
    boxes = build_scene(0)
    ds = VLADataSet(256, args["Ta"], args["size"], seed=777, boxes=boxes)

    print("[viz] trajectories ...")
    fig_trajectories(model, ds, boxes, a.out, a.n_start)
    print("[viz] action bars ...")
    bars = fig_action_bars(model, ds, a.out)
    print("[viz] world-model loop ...")
    nf = fig_world_model_loop(model, ds, boxes, a.out)
    with open(os.path.join(a.out, "bars.json"), "w", encoding="utf-8") as f:
        json.dump(bars, f, indent=2, ensure_ascii=False)
    print(f"[viz] saved to {a.out}  frames={nf}")


if __name__ == "__main__":
    main()
