# 迷你 VLA: π0 架构的视觉-语言-动作模型 (从零实现)

任务：3D 导航场景里"看画面、听指令、出下一步动作"。不依赖任何预训练权重，
2.38M 参数，纯 CPU 约 7 分钟训完，并验证语言指令对生成动作的可控性。
与[动作条件视频世界模型](https://github.com/yimingjiang216-alt/vision-worldmodel-projects)
共享同一套 3D 导航数据管线（`data.py` 逐字节共享），构成"世界模型生成轨迹 → VLA 学习策略"的闭环。

## 一、全景：从数据到结论的链条

```
data.py            渲染 3D 导航场景 (向量化光栅化, 不依赖渲染引擎)
   ↓
vla_data.py        生成训练对: 观测 48×48 + 6 类语言指令 → 未来 8 步动作真值
   ↓
model.py           2.38M VLA: MiniViT(视觉) + LangEncoder(语言) + FlowHead(动作)
   ↓
train.py           Flow Matching 训练 120 epoch, CPU 约 7 分钟
   ↓
evaluate.py        三项检验: 动作精度 / 语言可控性 / CFG 响应扫描
   ↓
结论               只改指令不改画面, 动作按指令反向 (响应 gap 0.96) —— 语言真正参与决策
```

## 二、一轮推理的数据流

每一步控制（动作 = 前进速度 v_forward + 角速度 ω，2 维，一次预测未来 8 步）：

```
观测图像 (3,48,48) ── PatchEmbed ──> 小 ViT (4层) ──┐
                                                     ├── 跨模态融合(1层Attn) ──> 条件 c
语言指令 (5 tokens) ── Embedding + 1层Attn ──────────┘            │
                                                                  v
噪声动作 a_0 ~ N(0,I) ──> 速度场 v_theta(a_t, t, c) <── 时间 t, FiLM 调制
                        Flow Matching: a_1 = a_0 + ∫ v dt  (ODE 积分 10 步)
                        -> 未来 8 步动作, 执行第一步, 环境前进, 进入下一轮
```

推理时 CFG 生效：v = v_null + cfg·(v − v_null)，cfg 越大语言影响越强。

## 三、为什么不用 π0/π0.5/π0.6/π0.7 的官方权重（选型）

| 模型 | 开源状态 | 硬件门槛 |
|---|---|---|
| π0 / π0-FAST / π0.5 | ✅ 权重开源 | 推理 >8GB 显存, 全量微调 >70GB (A100/H100), 仅 Ubuntu |
| π0.6 (RECAP) | ❌ 未开源(仅论文) | 还需真实机器人本体 + 人类干预数据 + RL 训练循环 |
| π0.7 (world-model) | ❌ 未开源(仅博客) | 第三方复现需 8 卡 GPU + flash-attn |

结论: 官方权重在本机(CPU, 无 GPU, Windows)无法训练, 因此从零复现架构,
训练与推理的全部实现都在本仓库内.

> **后记**: 2026-10 在 Kaggle 免费 T4 上跑通了真正的开源 π0.5 (LIBERO 微调权重, 3B)
> 驱动 LIBERO/robosuite 机械臂完成取物任务, 3/3 集成功, 详见
> [pi05-libero-sim](https://github.com/yimingjiang216-alt/pi05-libero-sim).
> 本仓库从零实现补架构理解, 那个仓库验证开源权重真机闭环 —— 同一套 π0 范式的两种尺度.

## 四、深度学习技术点

1. **Flow Matching** (Lipman et al. 2023): 训练目标 v* = a_1 - a_0,
   比 DDPM 的马尔可夫降噪更直接, 采样只需 10 步 ODE. 这是 π0 动作头的核心.
2. **Classifier-Free Guidance**: 训练时 15% 概率把语言条件置空,
   推理时 v = v_null + cfg * (v - v_null), cfg 越大语言影响越强.
3. **Action Chunking**: 一次预测 8 步未来动作, 减少自回归累积误差.
4. **FiLM 条件注入**: 时间 t 与视觉-语言条件 c 通过仿射变换调制动作网络.
5. **ViT 从零实现**: Patch Embedding + 位置编码 + Pre-Norm Block, 不依赖 timm.
6. **EMA + OneCycleLR + 梯度裁剪**: 标准训练稳定技巧.

## 五、方法与出处

| 本仓库用到的方法 | 出处 |
|---|---|
| π0 架构（VLA + flow matching 动作头 + action chunking） | Physical Intelligence, *π₀: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164（官方实现 Physical-Intelligence/openpi） |
| π0.5（见第三节后记的真机验证） | Physical Intelligence, arXiv:2504.16054 |
| Flow Matching | Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023, arXiv:2210.02747 |
| Classifier-Free Guidance | Ho & Salimans, *Classifier-Free Diffusion Guidance*, arXiv:2207.12598 |
| Action chunking | Zhao et al. (ACT), RSS 2023, arXiv:2304.13705 |
| ViT | Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021, arXiv:2010.11929 |
| FiLM 条件调制 | Perez et al., AAAI 2018, arXiv:1709.07871 |

## 六、结果

`python evaluate.py --ckpt ckpt/vla_best.pt`

| 指标 | 值 | 说明 |
|---|---|---|
| 动作 MSE | **0.0175** (val) / 0.155 (跨场景) | 预测动作 vs 真值 |
| 语言响应 gap | **0.959** | `turn left` (+0.481) vs `turn right` (-0.477) 的角速度差 |
| 语言有效性 | **true** | 判据 abs(gap) > 0.3, 模型确实"听指令" |
| CFG 响应 | 0.98 → 1.55 → 2.06 → 2.36 | cfg=1/2/4/8 下 left-right 差距单调递增 |

同一观测图像, 只改变语言指令, 生成的动作按指令改变
(左转/右转角速度符号相反): 语言条件参与了决策, 不是装饰性输入.

## 七、演示视频

| 视频 | 内容 |
|---|---|
| [demo_trajectories.mp4](videos/demo_trajectories.mp4) | 同一观测 + 5 种语言指令, 轨迹逐帧分化动画 |
| [demo_first_person.mp4](videos/demo_first_person.mp4) | 同一观测, `turn left` vs `turn right` 第一视角反向旋转对比 |

复现: `python make_video.py`(依赖 imageio / imageio-ffmpeg / matplotlib)

## 八、世界模型闭环

`visualize.py` 的 `world_model_loop` 用 VLA 生成的动作块驱动
世界模型项目的场景渲染器(`data.py` 的 `render_frame` + 同一动力学):

```
VLA (观测+指令 -> 动作)  ──动作──>  世界模型渲染器 (动作 -> 未来帧)
                                          |
                                    "想象未来" 33 帧
```

这条路径即"世界模型为 VLA 生成训练数据": 世界模型把动作转成未来观测,
VLA 根据观测+指令产出动作, 两者共享数据管线.

## 九、文件

| 文件 | 内容 |
|---|---|
| `data.py` | (复用自世界模型项目) 3D 导航场景 + 向量化光栅化渲染 |
| `vla_data.py` | VLA 数据集: 观测 + 6 类语言指令 -> 未来 8 步动作 |
| `model.py` | MiniViT + LangEncoder + FlowHead + CFG 采样 |
| `train.py` | Flow Matching 训练, 支持 --resume, EMA, OneCycleLR |
| `evaluate.py` | 动作精度 / 语言可控性 / CFG 响应扫描 |
| `visualize.py` | 轨迹图 / 动作对比图 / 世界模型闭环图 |

## 十、运行

Windows 一键跑全部演示(评估+出图+视频): 双击 `run_demo.bat` 或命令行执行.

```bash
python train.py --epochs 120          # 约 7 分钟 (CPU)
python evaluate.py --ckpt ckpt/vla_best.pt
python visualize.py --ckpt ckpt/vla_best.pt --out figs
```

## 十一、复现记录

- 初版 ODE 积分方向写反 (linspace 1->0), 导致采样动作发散到 MSE=201;
  修正为 0->1 后 MSE 降至 0.0175 (验证集).
- 初版空条件用零图像, 与训练时的 CFG dropout 分布不一致;
  改为"保留真实观测 + 置空语言"后 CFG sweep 恢复单调.
