# -*- coding: utf-8 -*-
"""Action-conditioned video world model: data generation.

生成合成"第一人称导航"视频片段: 智能体在 3D 场景中按动作序列移动,
每一帧由动作(a_t)决定下一时刻状态. 动作->状态的转移构成世界模型,
用于训练 action-conditioned video diffusion, 验证"给定动作序列生成未来视频".
"""
import numpy as np
import torch

TWO_PI = 2.0 * np.pi


def build_scene(seed=0, n_buildings=18, half=40.0):
    """程序化城市场景(与重建项目共用地形思路): 建筑盒 + 地面网格."""
    rng = np.random.default_rng(seed)
    H = np.zeros((int(2 * half), int(2 * half)), dtype=np.float32)
    boxes = []
    tries = 0
    while len(boxes) < n_buildings and tries < 400:
        tries += 1
        w = rng.uniform(5, 12); d = rng.uniform(5, 12)
        x = rng.uniform(-half + w, half - w); y = rng.uniform(-half + d, half - d)
        h = rng.uniform(10, 32)
        ok = True
        for (x0, y0, w0, d0, h0) in boxes:
            if abs(x - x0) < (w + w0) / 2 + 3 and abs(y - y0) < (d + d0) / 2 + 3:
                ok = False; break
        if ok:
            boxes.append((x, y, w, d, h))
    return boxes


def render_frame(state, boxes, W=64, Hh=64, half=40.0, height=6.0):
    """向量化光栅化: 从当前位姿渲染 RGB + 深度图(每条射线与所有建筑盒批量求交).

    state: (x, y, theta) 智能体水平位姿
    返回 (img, depth), img:(3,H,W) in [0,1], depth:(H,W) meters
    """
    x, y, th = state
    fx = fy = 60.0
    cx, cy = W / 2.0, Hh / 2.0
    fwd = np.array([np.cos(th), np.sin(th)], dtype=np.float32)
    right = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
    j = np.arange(W, dtype=np.float32) + 0.5
    i = np.arange(Hh, dtype=np.float32) + 0.5
    u = (j - cx) / fx                              # (W,)
    v = (Hh - 1 - i + 0.5 - cy) / fy               # (Hh,) 从下往上
    U, V = np.meshgrid(u, v)                       # (Hh, W)
    dz = V                                         # 相机系竖直分量
    dx = fwd[0] + U * right[0]                     # (Hh, W)
    dy = fwd[1] + U * right[1]

    img = np.zeros((3, Hh, W), dtype=np.float32)
    depth = np.full((Hh, W), 200.0, dtype=np.float32)
    sky = np.array([0.45, 0.62, 0.85], dtype=np.float32)

    below = dz < -1e-3                             # 朝下的射线才可能命中地面/建筑
    img[:, ~below] = sky[:, None]

    if below.any():
        dx_b = dx[below]; dy_b = dy[below]; dz_b = dz[below]
        t_ground = (-height) / dz_b                # 与水平面 y=0 交距
        gx = x + t_ground * dx_b; gy = y + t_ground * dy_b
        ok_g = (np.abs(gx) < half) & (np.abs(gy) < half) & (t_ground > 0)
        best_t = np.where(ok_g, t_ground, np.inf)

        # 与所有建筑盒批量求交
        hit_col = np.zeros((len(dx_b), 3), dtype=np.float32)
        for (bx, by, bw, bd, bh) in boxes:
            tx1 = (bx - bw / 2 - x) / np.where(np.abs(dx_b) > 1e-9, dx_b, 1e-9)
            tx2 = (bx + bw / 2 - x) / np.where(np.abs(dx_b) > 1e-9, dx_b, 1e-9)
            ty1 = (by - bd / 2 - y) / np.where(np.abs(dy_b) > 1e-9, dy_b, 1e-9)
            ty2 = (by + bd / 2 - y) / np.where(np.abs(dy_b) > 1e-9, dy_b, 1e-9)
            tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
            valid = (tmax > np.maximum(tmin, 0))
            t_hit = np.where(tmin > 0, tmin, tmax)
            # 建筑立面高度判据: 击中点竖直高度 z_hit 须落在 [0, bh]
            z_hit = -t_hit * dz_b                  # dz<0 -> 正高度
            hit = valid & (t_hit > 0) & (z_hit >= 0) & (z_hit <= bh)
            better = hit & (t_hit < best_t)
            if better.any():
                best_t = np.where(better, t_hit, best_t)
                c = 0.30 + 0.55 * (bh - 10) / 22.0
                # 简单明暗: 随高度渐变 + 朝向偏移
                shade = 0.85 + 0.25 * (z_hit / max(bh, 1e-6))
                col = np.array([c * 0.95, c * 0.78, c * 0.62],
                               dtype=np.float32)[:, None] * shade[None, :]
                hit_col[:, 0] = np.where(better, col[0], hit_col[:, 0])
                hit_col[:, 1] = np.where(better, col[1], hit_col[:, 1])
                hit_col[:, 2] = np.where(better, col[2], hit_col[:, 2])
        # 地面网格
        on_ground = (~np.isfinite(best_t)) | (best_t >= 1e8)
        ground_t = np.where(ok_g, t_ground, np.inf)
        on_ground = on_ground & (ground_t < 1e8) & (ground_t < best_t)
        if on_ground.any():
            gxg = x + ground_t * dx_b; gyg = y + ground_t * dy_b
            gx2 = np.mod(gxg + 200, 8.0); gy2 = np.mod(gyg + 200, 8.0)
            grid = np.where((np.minimum(gx2, 8 - gx2) < 0.35) |
                            (np.minimum(gy2, 8 - gy2) < 0.35), 0.85, 0.5)
            gcol = np.stack([grid * 0.72, grid * 0.70, grid * 0.62]).astype(np.float32)
            hit_col[:, 0] = np.where(on_ground, gcol[0], hit_col[:, 0])
            hit_col[:, 1] = np.where(on_ground, gcol[1], hit_col[:, 1])
            hit_col[:, 2] = np.where(on_ground, gcol[2], hit_col[:, 2])
            best_t = np.where(on_ground, ground_t, best_t)

        finite = np.isfinite(best_t)
        best_t = np.where(finite, best_t, 200.0)
        depth[below] = best_t
        img[:, below] = hit_col.T
    return img, depth


class ActionVideoDataset(torch.utils.data.Dataset):
    """动作条件视频数据集.

    动作空间 a = (v_forward, omega): 线速度/角速度.
    状态转移(世界模型动力学): x' = x + v*cos(th), y' = y + v*sin(th), th' = th + omega.
    每条样本: (video, action_seq), video:(T,3,H,W), action_seq:(T,2) 对齐到每帧.
    """

    def __init__(self, n_samples=512, T=8, size=64, seed=0, boxes=None):
        self.n = n_samples
        self.T = T
        self.size = size
        self.boxes = boxes if boxes is not None else build_scene(seed)
        self.seed = seed
        rng = np.random.default_rng(seed)
        # 预生成轨迹(渲染较慢, 缓存)
        self.cache = []
        for i in range(n_samples):
            self.cache.append(self._make_episode(rng))
        self.action_dim = 2

    def _make_episode(self, rng):
        """恒定转向速率航段: 每个 episode 采样一组常速(v, omega)并保持全程.

        这样"动作"与"视频"是确定性映射, 模型必须利用动作条件才能重建
        正确的转弯幅度与方向; 同时转向角速度直接决定画面的水平平移方向,
        便于用光流方向验证动作可控性.
        """
        half = 38.0
        x, y, th = 0.0, 0.0, rng.uniform(0, TWO_PI)
        v = rng.uniform(1.0, 4.0)                      # 恒速前进
        om = rng.uniform(-0.9, 0.9)                    # 恒定转向速率
        frames, acts = [], []
        for t in range(self.T):
            x = np.clip(x + v * np.cos(th), -half, half)
            y = np.clip(y + v * np.sin(th), -half, half)
            th = th + om
            img, dep = render_frame((x, y, th), self.boxes, self.size, self.size)
            frames.append(img)
            acts.append([v / 3.0, om / 0.9])           # 归一化到 ~[0.3,1.3] / [-1,1]
        return np.stack(frames), np.array(acts, dtype=np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        v, a = self.cache[i]
        return torch.from_numpy(v), torch.from_numpy(a)


def make_loaders(n_train=512, n_val=96, T=8, size=64, batch=32, seed=0):
    boxes = build_scene(seed)
    tr = ActionVideoDataset(n_train, T, size, seed, boxes)
    va = ActionVideoDataset(n_val, T, size, seed + 1000, boxes)
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True),
            torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False))