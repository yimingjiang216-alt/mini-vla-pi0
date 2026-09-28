# -*- coding: utf-8 -*-
"""迷你 VLA 评估: 语言条件有效性 + 动作精度.

核心问题: 模型生成的动作是否真的"听指令"? 评估三件事:

1. 动作精度: 预测动作 vs 真值的 MSE / 余弦方向一致性.
2. 语言可控性(关键): 同一观测, 换不同指令 -> 生成动作是否按指令改变?
   用"turn left" vs "turn right"的角速度差量化. 若模型只靠视觉回归,
   该差值应接近 0; 若语言真正条件化, 差值应显著.
3. CFG 响应: 扫 guidance 强度, 看动作如何随 cfg 移动.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import MiniVLA
import torch as _torch
from vla_data import (VLADataSet, encode_instruction, INSTRUCTIONS, VOCAB)


def enc_txt(text):
    """语言指令 -> LongTensor (1, MAX_LEN)."""
    return _torch.from_numpy(encode_instruction(text))


@torch.no_grad()
def eval_action_acc(model, ds, n=128, steps=10, cfg=1.0):
    model.eval()
    idx = np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)),
                                          replace=False)
    preds, gts, langs = [], [], []
    for i in idx:
        o, l, a = ds[int(i)]
        obs = o.unsqueeze(0); lang = l.unsqueeze(0)
        p = model.sample(obs, lang, steps=steps, cfg=cfg)
        preds.append(p[0].numpy()); gts.append(a.numpy())
        langs.append(l.numpy())
    P = np.stack(preds); G = np.stack(gts)
    mse = float(np.mean((P - G) ** 2))
    # 角速度分量(v[:,1])的符号一致率
    ps, gs = P[..., 1], G[..., 1]
    sign = float(np.mean(np.sign(ps) == np.sign(gs)))
    # 方向余弦
    cos = float(np.mean([np.dot(p.ravel(), g.ravel()) /
                         (np.linalg.norm(p) * np.linalg.norm(g) + 1e-8)
                         for p, g in zip(P, G)]))
    return {"action_mse": mse, "omega_sign_acc": sign, "cos_sim": cos}


@torch.no_grad()
def eval_language_control(model, ds, n=64, steps=10, cfg=1.0):
    """语言可控性: 固定观测, 只换指令, 测角速度差异.

    返回 left/right 指令下平均角速度的差值. 该值 > 阈值说明
    语言真正参与决策(而非装饰性输入).
    """
    model.eval()
    rng = np.random.default_rng(1)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    l_txt = enc_txt("turn left")
    r_txt = enc_txt("turn right")
    dl, dr = [], []
    for i in idx:
        o, _, a = ds[int(i)]
        obs = o.unsqueeze(0)
        pl = model.sample(obs, l_txt.unsqueeze(0), steps=steps, cfg=cfg)
        pr = model.sample(obs, r_txt.unsqueeze(0), steps=steps, cfg=cfg)
        dl.append(pl[0, :, 1].numpy())   # omega 分量
        dr.append(pr[0, :, 1].numpy())
    L, R = np.stack(dl), np.stack(dr)
    return {
        "omega_left": float(L.mean()), "omega_right": float(R.mean()),
        "omega_gap": float(L.mean() - R.mean()),
        "per_sample_abs_gap": float(np.abs(L - R).mean()),
    }


@torch.no_grad()
def eval_cfg_sweep(model, ds, n=48, steps=10):
    """扫 CFG 强度, 看语言条件如何放大动作差异."""
    model.eval()
    rng = np.random.default_rng(2)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    out = {}
    for cfg in [1.0, 2.0, 4.0, 8.0]:
        gaps = []
        for i in idx:
            o, _, _ = ds[int(i)]
            obs = o.unsqueeze(0)
            l_txt = enc_txt("turn left")
            r_txt = enc_txt("turn right")
            pl = model.sample(obs, l_txt.unsqueeze(0), steps=steps, cfg=cfg)
            pr = model.sample(obs, r_txt.unsqueeze(0), steps=steps, cfg=cfg)
            gaps.append(float((pl[0, :, 1] - pr[0, :, 1]).mean()))
        out[f"cfg_{cfg}"] = float(np.mean(np.abs(gaps)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt/best.pt")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--out", type=str, default="eval_results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    args = ck["args"]
    model = MiniVLA(img_size=args["size"], patch=8, dim=args["dim"],
                    depth=args["depth"], Ta=args["Ta"], vocab=len(VOCAB))
    sd = ck.get("ema", ck["model"])
    model.load_state_dict(sd)
    model.eval()

    ds = VLADataSet(256, args["Ta"], args["size"], seed=777,
                    boxes=__import__("data").build_scene(0))
    res = {
        "ckpt": a.ckpt,
        "train_epochs": ck["epoch"] + 1,
        "action": eval_action_acc(model, ds, a.n, a.steps),
        "language": eval_language_control(model, ds, 64, a.steps),
        "cfg_sweep": eval_cfg_sweep(model, ds, 48, a.steps),
    }
    # 足够强的语言响应判据: left/right 平均角速度差 > 0.3 (归一化尺度)
    res["language_effective"] = abs(res["language"]["omega_gap"]) > 0.3
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
