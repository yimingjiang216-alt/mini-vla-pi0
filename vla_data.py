# -*- coding: utf-8 -*-
"""Mini VLA 数据集: 观测图像 + 语言指令 -> 动作序列.

复用世界模型项目的 3D 导航合成场景(data.py), 把"视觉-语言-动作"
三元组构造出来, 供 π0 风格的 flow-matching VLA 训练.

样本结构:
  obs      : (3, H, W)        当前帧观测(3D 导航场景第一视角)
 指令在 VLA 里通常来自人类或 VLM; 这里用"指令模板 + 参数"合成,
          覆盖前进/转弯/速度档位, 形成可验证的语言-动作对应.
  action   : (Ta, 2)          未来 Ta 步动作 (v_forward, omega), 归一化
"""
import numpy as np
import torch
from data import build_scene, render_frame, TWO_PI

# 语言指令体系: (模板, 语义参数). 训练时模板转成 token id, 参数进入条件.
# 这套设计让"指令"真正影响最优动作 —— 转弯方向/速度档位由指令决定.
INSTRUCTIONS = [
    ("go straight",        {"omega": 0.0}),
    ("turn left",          {"omega": +0.55}),
    ("turn right",         {"omega": -0.55}),
    ("move forward fast",  {"v": 3.5}),
    ("move forward slowly", {"v": 1.2}),
    ("follow the road",    {"omega": 0.0, "v": 2.0}),
]
VOCAB = {w: i + 1 for i, w in enumerate(
    ["go", "straight", "turn", "left", "right", "move", "forward",
     "fast", "slowly", "follow", "the", "road", "<pad>"])}
VOCAB["<pad>"] = 0
MAX_LEN = 5


def encode_instruction(text):
    ids = [VOCAB.get(w, 0) for w in text.split()]
    ids = ids[:MAX_LEN] + [0] * (MAX_LEN - len(ids))
    return np.array(ids, dtype=np.int64)


class VLADataSet(torch.utils.data.Dataset):
    """Vision-Language-Action 数据集.

    每条样本: 给定当前观测 obs 与语言指令, 预测未来 Ta 步动作.
    指令决定轨迹的转弯方向与速度档位, 因此"正确动作"依赖于指令内容
    (这正是 VLA 的语言条件作用, 而非纯视觉回归).
    """

    def __init__(self, n_samples=1024, Ta=8, size=64, seed=0, boxes=None):
        self.n = n_samples
        self.Ta = Ta
        self.size = size
        self.boxes = boxes if boxes is not None else build_scene(seed)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.cache = []
        for _ in range(n_samples):
            self.cache.append(self._make(rng))

    def _make(self, rng):
        half = 38.0
        # 先采样"执行段": 从指令体系里选一条, 决定本段的 (v, omega)
        text, params = INSTRUCTIONS[rng.integers(len(INSTRUCTIONS))]
        v = params.get("v", rng.uniform(1.5, 3.0))
        om = params.get("omega", rng.uniform(-0.35, 0.35))

        # 从随机位姿出发, 先空跑几步进入场景, 取最后一帧作为"当前观测"
        x = rng.uniform(-20, 20); y = rng.uniform(-20, 20)
        th = rng.uniform(0, TWO_PI)
        for _ in range(rng.integers(3, 10)):
            x = np.clip(x + v * np.cos(th), -half, half)
            y = np.clip(y + v * np.sin(th), -half, half)
            th = th + om
        obs, _ = render_frame((x, y, th), self.boxes, self.size, self.size)

        # 再执行 Ta 步, 收集"未来动作" —— 动作本身由指令参数决定
        acts = []
        for _ in range(self.Ta):
            x = np.clip(x + v * np.cos(th), -half, half)
            y = np.clip(y + v * np.sin(th), -half, half)
            th = th + om
            acts.append([v / 3.0, om / 0.9])
        action = np.array(acts, dtype=np.float32)

        lang = encode_instruction(text)
        return obs.astype(np.float32), lang, action

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        o, l, a = self.cache[i]
        return torch.from_numpy(o), torch.from_numpy(l), torch.from_numpy(a)


def make_loaders(n_train=1024, n_val=192, Ta=8, size=64, batch=64, seed=0):
    boxes = build_scene(seed)
    tr = VLADataSet(n_train, Ta, size, seed, boxes)
    va = VLADataSet(n_val, Ta, size, seed + 500, boxes)
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True),
            torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False))


if __name__ == "__main__":
    tr, va = make_loaders(64, 16, Ta=4, size=48)
    o, l, a = next(iter(tr))
    print("obs", o.shape, o.dtype, "min", o.min().item(), "max", o.max().item())
    print("lang", l.shape, l[:2].tolist())
    print("action", a.shape, a.min().item(), a.max().item())
