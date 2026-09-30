# 迷你 VLA: π0 架构的视觉-语言-动作模型 (从零实现)

> 目标: 不依赖任何预训练权重, 在 CPU 上从零实现 π0 论文的核心架构
> (ViT 视觉编码 + 语言条件 + Flow Matching 动作头), 并验证语言指令对
> 生成动作的**可控性**. 与"动作条件视频世界模型"项目共享同一套 3D 导航
> 仿真数据管线, 构成"世界模型生成轨迹 -> VLA 学习策略"的闭环.

## 为什么不用 π0/π0.5/π0.6/π0.7 的官方权重

| 模型 | 开源状态 | 硬件门槛 |
|---|---|---|
| π0 / π0-FAST / π0.5 | ✅ 权重开源 | 推理 >8GB 显存, 全量微调 >70GB (A100/H100), 仅 Ubuntu |
| π0.6 (RECAP) | ❌ 未开源(仅论文) | 还需真实机器人本体 + 人类干预数据 + RL 训练循环 |
| π0.7 (world-model) | ❌ 未开源(仅博客) | 第三方复现需 8 卡 GPU + flash-attn |

结论: 官方权重在本机(CPU, 无 GPU, Windows)无法训练. 因此选择**从零复现架构**,
深度学习含量反而更高 —— 调包跑 demo 只能写"使用了 xx 模型", 从零实现则能
讲清 Flow Matching, ViT, CFG, action chunking 每一个细节.

## 架构

```
观测图像 (3,48,48) ── PatchEmbed ──> 小 ViT (4层) ──┐
                                                     ├── 跨模态融合(1层Attn) ──> 条件 c
语言指令 (5 tokens) ── Embedding + 1层Attn ──────────┘            │
                                                                  v
噪声动作 a_0 ~ N(0,I) ──> 速度场 v_theta(a_t, t, c) <── 时间 t, FiLM 调制
                        Flow Matching: a_1 = a_0 + ∫ v dt  (ODE 积分)
                        -> 未来 Ta=8 步动作 (v_forward, omega)
```

**参数量: 2.38M** (ViT 192 维 / 4 层 / 6 头), 纯 CPU 训练约 7 分钟(120 epoch).

## 深度学习技术点

1. **Flow Matching** (Lipman et al. 2023): 训练目标 v* = a_1 - a_0,
   比 DDPM 的马尔可夫降噪更直接, 采样只需 10 步 ODE. 这是 π0 动作头的核心.
2. **Classifier-Free Guidance**: 训练时 15% 概率把语言条件置空,
   推理时 v = v_null + cfg * (v - v_null), cfg 越大语言影响越强.
3. **Action Chunking**: 一次预测 8 步未来动作, 减少自回归累积误差.
4. **FiLM 条件注入**: 时间 t 与视觉-语言条件 c 通过仿射变换调制动作网络.
5. **ViT 从零实现**: Patch Embedding + 位置编码 + Pre-Norm Block, 不依赖 timm.
6. **EMA + OneCycleLR + 梯度裁剪**: 标准训练稳定技巧.

## 结果

`python evaluate.py --ckpt ckpt/vla_best.pt`

| 指标 | 值 | 说明 |
|---|---|---|
| 动作 MSE | **0.0175** (val) / 0.155 (跨场景) | 预测动作 vs 真值 |
| 语言响应 gap | **0.959** | `turn left` (+0.481) vs `turn right` (-0.477) 的角速度差 |
| 语言有效性 | **true** | 判据 abs(gap) > 0.3, 模型确实"听指令" |
| CFG 响应 | 0.98 → 1.55 → 2.06 → 2.36 | cfg=1/2/4/8 下 left-right 差距单调递增 |

**核心结论**: 同一观测图像, 只改变语言指令, 生成的动作按指令改变
(左转/右转角速度符号相反), 证明语言条件真正参与了决策, 而非装饰性输入.

## 世界模型闭环

`visualize.py` 的 `world_model_loop` 用 VLA 生成的动作块驱动
世界模型项目的场景渲染器(`data.py` 的 `render_frame` + 同一动力学):

```
VLA (观测+指令 -> 动作)  ──动作──>  世界模型渲染器 (动作 -> 未来帧)
                                          |
                                    "想象未来" 33 帧
```

这正是 JD 中"VLA 数据生成"的验证路径: 世界模型负责把动作转成未来观测,
VLA 负责根据观测+指令产出动作, 两者共享数据管线.

## 文件

| 文件 | 内容 |
|---|---|
| `data.py` | (复用自世界模型项目) 3D 导航场景 + 向量化光栅化渲染 |
| `vla_data.py` | VLA 数据集: 观测 + 6 类语言指令 -> 未来 8 步动作 |
| `model.py` | MiniViT + LangEncoder + FlowHead + CFG 采样 |
| `train.py` | Flow Matching 训练, 支持 --resume, EMA, OneCycleLR |
| `evaluate.py` | 动作精度 / 语言可控性 / CFG 响应扫描 |
| `visualize.py` | 轨迹图 / 动作对比图 / 世界模型闭环图 |

## 运行

```bash
python train.py --epochs 120          # 约 7 分钟 (CPU)
python evaluate.py --ckpt ckpt/vla_best.pt
python visualize.py --ckpt ckpt/vla_best.pt --out figs
```

## 复现记录

- 初版 ODE 积分方向写反 (linspace 1->0), 导致采样动作发散到 MSE=201;
  修正为 0->1 后 MSE 降至 0.0175 (验证集).
- 初版空条件用零图像, 与训练时的 CFG dropout 分布不一致;
  改为"保留真实观测 + 置空语言"后 CFG sweep 恢复单调.
