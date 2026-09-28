# -*- coding: utf-8 -*-
"""迷你 VLA 训练: Flow Matching + Classifier-Free Guidance.

训练目标(π0 的核心 loss):
    给真值动作 a_1, 采样 a_0 ~ N(0,I) 与 t ~ U(0,1),
    目标速度场 v* = a_1 - a_0  (直线流, OT-CM),
    最小化 || v_theta(a_t, t, c) - v* ||^2, 其中 a_t = (1-t)a_0 + t a_1.

CFG dropout: 以概率 p 随机把条件换成"空条件", 让模型同时学到
条件分布与边缘分布, 推理时才能做 classifier-free guidance.

用法: python train.py --epochs 60
支持 --resume 断点续训.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import MiniVLA
from vla_data import make_loaders, VOCAB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--n_val", type=int, default=192)
    p.add_argument("--Ta", type=int, default=8)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cfg_dropout", type=float, default=0.15)
    p.add_argument("--out", type=str, default="ckpt")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    tr, va = make_loaders(a.n_train, a.n_val, a.Ta, a.size, a.batch, a.seed)
    model = MiniVLA(img_size=a.size, patch=8, dim=a.dim, depth=a.depth,
                    Ta=a.Ta, vocab=len(VOCAB)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[mini-vla] params: {n_par/1e6:.2f}M  device: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(tr.dataset) / a.batch)
    total_steps = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.1)
    ema = {k: v.detach().clone().float()
           for k, v in model.state_dict().items()}

    start_ep = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.1,
            last_step=start_ep * steps_per_epoch)
        ema = ck.get("ema", ema)
        print(f"[mini-vla] resumed from {a.resume} @ epoch {start_ep}")

    hist = []
    best = float("inf")
    null_lang = torch.zeros(1, 5, dtype=torch.long)

    for ep in range(start_ep, a.epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for obs, lang, act in tr:
            obs, lang, act = obs.to(device), lang.to(device), act.to(device)
            B = obs.size(0)
            # CFG dropout: 随机置空条件(语言+视觉语义)
            keep = torch.rand(B) > a.cfg_dropout
            lang_c = torch.where(keep[:, None], lang, null_lang.expand(B, -1))
            cond = model.encode(obs, lang_c)

            a1 = act
            a0 = torch.randn_like(a1)
            t = torch.rand(B)
            at = (1 - t[:, None, None]) * a0 + t[:, None, None] * a1
            target = a1 - a0
            v = model.velocity(at, t, cond)
            loss = F.mse_loss(v, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                d = 0.999
                for k, v_ in model.state_dict().items():
                    if v_.dtype.is_floating_point:
                        ema[k].mul_(d).add_(v_.detach(), alpha=1 - d)
                    else:
                        ema[k] = v_.detach().clone()
            tot += loss.item(); nb += 1
        tr_loss = tot / max(nb, 1)

        # 验证: 直接动作 MSE 与流匹配 loss
        model.eval()
        vtot, vnb, mtot, mnb = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for obs, lang, act in va:
                obs, lang, act = obs.to(device), lang.to(device), act.to(device)
                B = obs.size(0)
                cond = model.encode(obs, lang)
                a1, a0 = act, torch.randn_like(act)
                t = torch.rand(B)
                at = (1 - t[:, None, None]) * a0 + t[:, None, None] * a1
                vtot += F.mse_loss(model.velocity(at, t, cond), a1 - a0).item()
                vnb += 1
                # 真实动作误差(采样 5 步 ODE)
                pred = model.sample(obs, lang, steps=5)
                mtot += F.mse_loss(pred, act).item()
                mnb += 1
        va_flow, va_mse = vtot / max(vnb, 1), mtot / max(mnb, 1)
        hist.append({"epoch": ep, "train": tr_loss, "val_flow": va_flow,
                     "val_action_mse": va_mse, "sec": time.time() - t0})
        if va_mse < best:
            best = va_mse
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema, "epoch": ep, "args": vars(a),
                        "val_action_mse": va_mse},
                       os.path.join(a.out, "best.pt"))
        if (ep + 1) % a.log_every == 0 or ep == a.epochs - 1:
            print(f"ep {ep+1:3d}/{a.epochs}  tr {tr_loss:.4f}  "
                  f"val_flow {va_flow:.4f}  actMSE {va_mse:.4f}  "
                  f"best {best:.4f}  {hist[-1]['sec']:.1f}s", flush=True)

    torch.save({"model": model.state_dict(), "ema": ema, "epoch": a.epochs - 1,
                "args": vars(a), "val_action_mse": best},
               os.path.join(a.out, "last.pt"))
    with open(os.path.join(a.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print(f"[mini-vla] done. best action MSE = {best:.4f}")


if __name__ == "__main__":
    main()
