# -*- coding: utf-8 -*-
"""π0 风格的迷你 VLA 模型: 视觉-语言条件 + Flow Matching 动作生成.

架构(严格按 π0 论文的"条件流匹配"思想, 缩小到 CPU 可训规模):

  观测图像 -> Patch Embedding -> 小 ViT -> 视觉 token
  语言指令 -> Token Embedding -> 线性投影 -> 语言 token
      两路 token 拼接 -> 条件表征 c  (跨模态融合, π0 里由 VLM backbone 完成)
      噪声动作 a_1 ~ N(0,I) -> 速度场 v_theta(a_t, t, c)
      Flow Matching 训练: 学从噪声到真值动作的速度场
      推理: ODE 积分 N 步, 从噪声"流"到干净动作

深度学习要点:
  - Flow Matching (Lipman et al. 2023): 比 DDPM 扩散更省步数, π0 的动作头
  - Classifier-Free Guidance: 训练时随机 dropout 条件, 推理时放大条件方向
  - Action Chunking: 一次预测 Ta 步动作, 减少累积误差
  - 线速度/角速度归一化, 与世界模型项目共用地面动力学
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """图像 -> patch token, 纯卷积实现, 无需依赖 timm."""

    def __init__(self, img_size=48, patch=8, in_ch=3, dim=192):
        super().__init__()
        assert img_size % patch == 0
        self.n_tok = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        h = self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)
        return h + self.pos


class Block(nn.Module):
    """标准 Pre-Norm Transformer Block (自注意力 + MLP)."""

    def __init__(self, dim, heads=6, mlp_ratio=2.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class MiniViT(nn.Module):
    """视觉编码器: 纯手写 ViT, 输出 CLS + patch token."""

    def __init__(self, img_size=48, patch=8, dim=192, depth=4, heads=6):
        super().__init__()
        self.embed = PatchEmbed(img_size, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.embed(x)
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1)
        for b in self.blocks:
            h = b(h)
        return self.norm(h)


class LangEncoder(nn.Module):
    """语言编码: 小词表嵌入 + 一层自注意力 + 平均池化 -> 条件向量."""

    def __init__(self, vocab=14, dim=192, max_len=5):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.block = Block(dim, heads=4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids):
        m = ids != 0
        h = self.emb(ids) + self.pos[:, : ids.size(1)]
        h = self.block(h)
        h = self.norm(h)
        h = h * m.unsqueeze(-1)
        return h.sum(1) / m.sum(1, keepdim=True).clamp(min=1)


class FlowHead(nn.Module):
    """速度场 v_theta(a_t, t, c): 用 MLP + FiLM 条件注入.

    输入: 噪声动作 a_t (B, Ta, A), 时间 t (B,), 条件 c (B, D)
    输出: 速度 (B, Ta, A), 指向真值动作的方向.
    """

    def __init__(self, action_dim=2, Ta=8, cond_dim=192, hidden=256):
        super().__init__()
        self.Ta = Ta
        self.action_dim = action_dim
        self.a_emb = nn.Sequential(nn.Linear(action_dim, hidden), nn.SiLU())
        self.t_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        # FiLM: 时间调制动作特征
        self.film_t = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden))
        self.film_c = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden))
        self.body = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, a_t, t, c):
        B = a_t.size(0)
        h = self.a_emb(a_t)
        tt = self.t_mlp(t.view(B, 1))
        h = h * (1 + self.film_t(tt))[:, None] + self.film_c(c)[:, None]
        return self.out(self.body(h))


class MiniVLA(nn.Module):
    """迷你 π0: ViT + 语言编码 融合成条件, Flow Matching 生成动作块."""

    def __init__(self, img_size=48, patch=8, dim=192, depth=4, heads=6,
                 action_dim=2, Ta=8, vocab=14, lang_len=5):
        super().__init__()
        self.vision = MiniViT(img_size, patch, dim, depth, heads)
        self.lang = LangEncoder(vocab, dim, lang_len)
        # 跨模态融合: 视觉 CLS 与语言向量双向注意力(一层)后取 CLS 作条件
        self.fuse = Block(dim, heads)
        self.v_proj = nn.Linear(dim, dim)
        self.flow = FlowHead(action_dim, Ta, dim)
        self.Ta = Ta
        self.action_dim = action_dim

    def encode(self, obs, lang_ids):
        v = self.vision(obs)                       # (B, 1+N, D)
        l = self.lang(lang_ids).unsqueeze(1)       # (B, 1, D)
        fused = self.fuse(torch.cat([l, v], dim=1))
        return self.v_proj(fused[:, 0])            # 语言位作 CLS

    def velocity(self, a_t, t, cond):
        return self.flow(a_t, t, cond)

    @torch.no_grad()
    def sample(self, obs, lang_ids, steps=10, cfg=1.0, cond_drop=0.0):
        """ODE 积分采样: 从高斯噪声流到动作. steps 越多越准但越慢.

        cfg: classifier-free guidance 强度; 1.0 = 不引导.
        训练时见过的条件被随机 dropout, 这里用"空条件"作负样本.
        """
        B = obs.size(0)
        device = next(self.parameters()).device
        a = torch.randn(B, self.Ta, self.action_dim, device=device)
        cond = self.encode(obs, lang_ids)
        # 空条件须与训练时的 CFG dropout 一致: 保留真实观测, 只置空语言
        null_lang = torch.zeros_like(lang_ids)
        cond_null = self.encode(obs, null_lang)
        # 时间正向 0 -> 1: 从高斯噪声"流"到真值动作
        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            tt = torch.full((B,), t0.item(), device=device)
            v = self.velocity(a, tt, cond)
            if cfg != 1.0:
                v_null = self.velocity(a, tt, cond_null)
                v = v_null + cfg * (v - v_null)
            a = a + (t1 - t0) * v                  # 欧拉积分
        return a


if __name__ == "__main__":
    m = MiniVLA(img_size=48, Ta=4)
    n = sum(p.numel() for p in m.parameters())
    obs = torch.randn(4, 3, 48, 48)
    lang = torch.randint(0, 14, (4, 5))
    cond = m.encode(obs, lang)
    a = torch.randn(4, 4, 2)
    v = m.velocity(a, torch.rand(4), cond)
    s = m.sample(obs, lang, steps=5)
    print(f"params: {n/1e6:.2f}M  cond {tuple(cond.shape)} vel {tuple(v.shape)}")
    print("sample", tuple(s.shape), "range", s.min().item(), s.max().item())
