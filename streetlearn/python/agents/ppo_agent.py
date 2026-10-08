# Copyright 2026 The StreetLearn Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language permissions and
# limitations under the License.

"""PyTorch PPO 策略/价值网络 —— goal / policy 双 LSTM，连接方式可选。

============================================================================
两种连接方式（--model_topology）
============================================================================
官方 CityNav 的实际前向（city_nav_agent.py 的 _core）是**串联**：goal LSTM
处理视觉与目标，输出经 Dropout 和 64 维 tanh 瓶颈送进 policy LSTM；policy LSTM
再结合视觉、上一动作和奖励做决策，策略头 / 价值头只读 policy LSTM 的输出。
（论文文字、基类 GoalNav 与 CityNav 覆写后的前向路径不同，以 _core 为准。）

  parallel（默认；当前并联实验结构，旧 checkpoint 都是它）：
    goal_lstm  ：cat(view_feat, goal_feat, state_feat, r_{t-1}, a_{t-1}) -> gh
    policy_lstm：cat(view_feat, state_feat, r_{t-1}, a_{t-1})             -> ph
    双头读 cat(gh, ph)（2×hidden）。policy LSTM 既看不到目标，也收不到 gh。

  serial（串联对照实验：只改连接，容量和输入不变）：
    goal_lstm 同上 -> gh；goal_message = tanh(Linear(gh -> 64))（首轮不加 Dropout）
    policy_lstm：cat(view_feat, state_feat, r_{t-1}, a_{t-1}, goal_message) -> ph
    双头只读 ph（hidden）。ResNet18 下 policy 输入 582 -> 646 维，头输入 1024 -> 512 维。

  serial 不是完整复现官方网络：官方每路 LSTM 256 维，视觉网络和输入也不同。
  一次改太多就分不清收益来自连接还是容量，所以结构对比只改连接。

两种结构的参数形状不同：checkpoint 记录 model_topology，评估 / 轨迹可视化按它
重建网络（agent_kwargs_from_checkpoint）；旧 checkpoint 没有该字段，按 parallel 加载。

============================================================================
输入（agent 不看底层图）与遮蔽开关
============================================================================
  view_image  [B,3,H,W]  第一人称街景；底部 status_height 行是邻居方向栏
  scalar      [B,6]       = sin(yaw), cos(yaw), latlng(2), target_latlng(2)
  prev_reward [B,1]       上一步奖励 r_{t-1}（裁剪到 [-1,1]）
  prev_action [B,A]       上一步动作 a_{t-1} 的 one-hot

  自身朝向 / 位置经 StateEmbed 送进两路 LSTM；目标经纬度经 GoalEmbed 只进 goal LSTM。
  官方 CityNav 策略不直接读自身位姿，训练时也不画方向栏（status_height=0）。
  为区分这些额外信息带来的收益，三类信息可以分别遮蔽（张量形状不变，目标始终保留）：
    mask_yaw          scalar[:, 0:2] 置 0（固定的"不给朝向"输入）
    mask_position     scalar[:, 2:4] 置 0
    mask_status_rows  图像底部这么多行固定为黑色（等价于归一化前置 0），上方投影区不动
  遮蔽在网络内部完成：采样、序列更新、bootstrap 估值、评估、轨迹可视化走的是同一处，
  设置随 checkpoint 保存。辅助 heading 任务的目标仍是真实朝向（取自未遮蔽的 scalar）。

  辅助 heading 头（--aux_heading）：从 goal LSTM 输出预测 (sinθ, cosθ)。

  目标表示（差异 9）：--goal_type target_latlng（默认）/ target_landmarks（未接入）。
  视觉编码器：--vision_backbone resnet18（默认）/ small_cnn（论文 84×84 小卷积）。
============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 支持的目标表示；target_landmarks 仅占位、默认不启用（差异 9）。
SUPPORTED_GOAL_TYPES = ("target_latlng", "target_landmarks")
# 双 LSTM 的连接方式（见模块说明）。
SUPPORTED_TOPOLOGIES = ("parallel", "serial")
# 串联结构里 goal -> policy 的瓶颈维数（官方 CityNav 同为 64）。
GOAL_MESSAGE_DIM = 64

# ImageNet 归一化常数：obs_to_batch 用它归一化图像，方向栏遮蔽用它算黑色的归一化值。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# checkpoint 里描述网络结构的字段及旧 checkpoint 缺省时的取值（续训比对、
# 评估 / 轨迹可视化重建网络都按这一份规则）。
MODEL_META_DEFAULTS = {
    "hidden_size": 512,
    "aux_heading": False,
    "vision_backbone": "resnet18",
    "goal_type": "target_latlng",
    "model_topology": "parallel",
    "mask_position": False,
    "mask_yaw": False,
    "mask_status_rows": 0,
}

# [P0-2 已废弃] 旧版按 1/REWARD_SCALE 压缩 prev_reward。现对齐论文语义：
# "Rewards are clipped at 1 for the purpose of gradient calculations"——
# 进入网络的上一步奖励同样裁剪到 [-1, 1]（与 GAE 侧统一）。常量保留仅为
# 向后兼容（旧 checkpoint 的文档引用），不再参与计算。
REWARD_SCALE = 100.0


class VisionEncoder(nn.Module):
    """视觉编码器（可切换骨干）。

    - resnet18：去分类 FC，输出 512（默认，有意增强，参数量大、特征好）。
    - small_cnn：论文风格小卷积（类 DQN/A3C），输出 512 以匹配 hidden_size。
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()
        if backbone == "resnet18":
            from torchvision import models
            resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.body = nn.Sequential(*list(resnet.children())[:-1])
            self.out_dim = 512
        elif backbone == "small_cnn":
            # 论文风格：两层卷积 + 自适应池化 + 线性映射到 512。
            self.body = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=8, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.fc = nn.Linear(32 * 7 * 7, 512)
            self.out_dim = 512
        else:
            raise ValueError("unknown vision_backbone: %s" % backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(x)              # [B, 512, 1, 1] 或 [B, 32, 7, 7]
        if hasattr(self, "fc"):
            f = self.fc(f.flatten(1))  # small_cnn：映射到 512
        return f.flatten(1)           # [B, 512]


class GoalEmbed(nn.Module):
    """目标描述符通路（论文的 goal descriptor g_t）。

    默认 target_latlng：把目标经纬度 (2 维) 嵌入为 goal_feat (64)。
    target_landmarks 表示预留接口（当前未接入，差异 9）。
    """

    def __init__(self, goal_dim: int = 2, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(goal_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, feat_dim]


class StateEmbed(nn.Module):
    """状态通路：把 [sin(yaw), cos(yaw), 当前 latlng(2)] (4 维) 嵌入为 state_feat (64)。

    论文靠第一人称图像隐式携带朝向/位置；这里显式给出（可用 mask_yaw /
    mask_position 遮蔽，检验这部分额外信息的贡献）。
    """

    def __init__(self, state_dim: int = 4, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, feat_dim]


class PPOAgent(nn.Module):
    """循环（recurrent）策略/价值网络：goal / policy 双 LSTM，并联或串联。"""

    def __init__(
        self,
        num_actions: int,
        hidden_size: int = 512,
        aux_heading: bool = False,
        pretrained_vision: bool = False,
        vision_backbone: str = "resnet18",
        goal_type: str = "target_latlng",
        model_topology: str = "parallel",
        mask_position: bool = False,
        mask_yaw: bool = False,
        mask_status_rows: int = 0,
    ):
        super().__init__()
        if goal_type not in SUPPORTED_GOAL_TYPES:
            raise ValueError(
                "goal_type must be one of %s, got %r" % (SUPPORTED_GOAL_TYPES, goal_type))
        if model_topology not in SUPPORTED_TOPOLOGIES:
            raise ValueError("model_topology must be one of %s, got %r"
                             % (SUPPORTED_TOPOLOGIES, model_topology))
        if int(mask_status_rows) < 0:
            raise ValueError("mask_status_rows must be >= 0, got %r" % mask_status_rows)
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.aux_heading = aux_heading
        self.goal_type = goal_type
        self.vision_backbone = vision_backbone
        self.model_topology = model_topology
        self.mask_position = bool(mask_position)
        self.mask_yaw = bool(mask_yaw)
        self.mask_status_rows = int(mask_status_rows)

        # 视觉编码器（共享卷积主干，可切换 resnet18 / small_cnn）。
        self.vision = VisionEncoder(backbone=vision_backbone,
                                    pretrained=pretrained_vision)

        if goal_type == "target_landmarks":
            # 差异 9：landmark 表示"可选但不启用"——接口已留，数据/嵌入未接。
            raise NotImplementedError(
                "目标表示 'target_landmarks' 尚未接入 PPO 版（需要 landmark 嵌入网络"
                "与对应数据集字段）。当前仅 'target_latlng' 可用，请用"
                " --goal_type target_latlng。")

        # 目标描述符通路：target_latlng(2) -> goal_feat(64)。
        self.goal_embed = GoalEmbed(goal_dim=2)
        # 状态通路：yaw sin/cos + 当前 latlng(2) -> state_feat(64)。
        self.state_embed = StateEmbed(state_dim=4)

        vd = self.vision.out_dim        # 512
        gd = self.goal_embed.out_dim     # 64
        sd = self.state_embed.out_dim    # 64

        # goal LSTM 两种结构相同：视觉 + 目标 + 自身状态 + r_{t-1} + a_{t-1}。
        goal_in = vd + gd + sd + 1 + num_actions
        self.goal_lstm = nn.LSTM(goal_in, hidden_size, num_layers=1,
                                 batch_first=False)
        if model_topology == "serial":
            # 串联：goal LSTM 输出经 64 维 tanh 瓶颈每步送进 policy LSTM；
            # 双头只读 policy LSTM。
            self.goal_to_policy = nn.Sequential(
                nn.Linear(hidden_size, GOAL_MESSAGE_DIM), nn.Tanh())
            policy_in = vd + sd + 1 + num_actions + GOAL_MESSAGE_DIM
            head_in = hidden_size
        else:
            # 并联：policy LSTM 不接目标，双头读两路输出的拼接。
            policy_in = vd + sd + 1 + num_actions
            head_in = 2 * hidden_size
        self.policy_lstm = nn.LSTM(policy_in, hidden_size, num_layers=1,
                                   batch_first=False)
        self.policy_head = nn.Linear(head_in, num_actions)
        self.value_head = nn.Linear(head_in, 1)

        # 辅助 heading 头（可选）：从 goal_hidden 预测 (sinθ, cosθ)。
        if aux_heading:
            self.heading_head = nn.Linear(hidden_size, 2)

        # 方向栏遮蔽的填充值：黑色像素（归一化前为 0）按 ImageNet 归一化后的值，
        # 与 obs_to_batch 对 0 像素的计算逐位一致。不进 state_dict，新旧
        # checkpoint 都能严格加载。
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("status_fill", (0.0 - mean) / std, persistent=False)

    def initial_state(self, batch_size: int = 1, device=None):
        """返回全零的双 LSTM 初始状态 (hp, cp, hg, cg)，形状均为 [1, B, hidden]。"""
        z = torch.zeros(1, batch_size, self.hidden_size,
                        dtype=torch.float32, device=device)
        return z, z.clone(), z.clone(), z.clone()  # (hp, cp, hg, cg)

    def mask_inputs(self, view, scalar):
        """按遮蔽设置处理输入，返回 (view, state_scalar)。

        state_scalar = scalar[:, 0:4]（yaw sin/cos + 当前 latlng）。目标经纬度
        scalar[:, 4:6] 不经过这里，_encode 直接读取，始终保留。
        不改调用方的张量：需要遮蔽时先 clone。
        """
        state = scalar[:, 0:4]
        if self.mask_yaw or self.mask_position:
            state = state.clone()
            if self.mask_yaw:
                state[:, 0:2] = 0.0
            if self.mask_position:
                state[:, 2:4] = 0.0
        if self.mask_status_rows:
            view = view.clone()
            view[:, :, -self.mask_status_rows:, :] = self.status_fill
        return view, state

    def _encode(self, view, scalar):
        """视觉 + 目标 + 状态 三路编码（遮蔽在这里统一生效）。

        返回 (view_f [B,512], goal_f [B,64], state_f [B,64])。
        scalar 固定布局：[0:2]=yaw sin/cos, [2:4]=当前 latlng, [4:6]=目标 latlng。
        """
        view, state_scalar = self.mask_inputs(view, scalar)
        view_f = self.vision(view)                    # [B, 512]
        goal_f = self.goal_embed(scalar[:, 4:6])      # [B, 64] 目标经纬度（不遮蔽）
        state_f = self.state_embed(state_scalar)      # [B, 64] 当前位姿
        return view_f, goal_f, state_f

    def forward(self, view, scalar, prev_reward, prev_action, hp, cp, hg, cg,
                dones=None):
        """单步前向（seq_len=1）。

        Args:
          view:        [B, 3, H, W] float，已做 ImageNet 归一化。
          scalar:      [B, 6] float（yaw sin/cos + latlng + target_latlng）。
          prev_reward: [B] 或 [B,1] float，上一步奖励 r_{t-1}（原值，内部 clamp 到 [-1,1]）。
          prev_action: [B] long，上一步动作 a_{t-1}（用于 one-hot）。
          hp,cp: policy_lstm 隐/细胞状态 [1, B, hidden]。
          hg,cg: goal_lstm  隐/细胞状态 [1, B, hidden]。

        Returns（调用者必须按此顺序解包；两种结构相同）:
          logits: [B, num_actions]   策略头原始输出
          value:  [B, 1]             价值头状态价值估计
          hp,cp:  更新后的 policy_lstm 状态
          hg,cg:  更新后的 goal_lstm 状态
          aux:    [B, 2] 预测 (sinθ,cosθ)；若关闭 aux_heading 则为 None
        """
        # DDP 只能跟踪 module(...) 的 forward，序列 PPO 更新也经此入口。
        if dones is not None:
            return self.forward_seq(view, scalar, prev_reward, prev_action,
                                    dones, hp, cp, hg, cg)
        view_f, goal_f, state_f = self._encode(view, scalar)
        # [P0-2] 论文语义：进入网络的上一步奖励裁剪到 [-1,1]（与 GAE 侧统一，
        # 替换旧的 /REWARD_SCALE —— 两处不一致曾是价值损失爆量的根因之一）。
        pr = prev_reward.view(-1, 1).to(view_f.dtype).clamp_(-1.0, 1.0)  # [B, 1]
        pa = F.one_hot(prev_action.long().view(-1),
                       self.num_actions).to(view_f.dtype)  # [B, A]

        # ---- goal LSTM ----
        g_in = torch.cat([view_f, goal_f, state_f, pr, pa], dim=1).unsqueeze(0)
        gh, (hg, cg) = self.goal_lstm(g_in, (hg, cg))
        gh = gh.squeeze(0)                                 # [B, hidden]
        # ---- policy LSTM ----
        if self.model_topology == "serial":
            # 串联：policy LSTM 每步读 goal LSTM 的瓶颈输出，双头只读 policy。
            goal_message = self.goal_to_policy(gh)         # [B, 64]
            p_in = torch.cat([view_f, state_f, pr, pa, goal_message],
                             dim=1).unsqueeze(0)
            ph, (hp, cp) = self.policy_lstm(p_in, (hp, cp))
            head_in = ph.squeeze(0)                        # [B, hidden]
        else:
            # 并联：两路输出拼接后给双头。
            p_in = torch.cat([view_f, state_f, pr, pa], dim=1).unsqueeze(0)
            ph, (hp, cp) = self.policy_lstm(p_in, (hp, cp))
            head_in = torch.cat([gh, ph.squeeze(0)], dim=1)  # [B, 2*hidden]
        logits = self.policy_head(head_in)                 # [B, num_actions]
        value = self.value_head(head_in)                   # [B, 1]
        aux = self.heading_head(gh) if self.aux_heading else None  # [B, 2]
        return logits, value, hp, cp, hg, cg, aux

    def get_value(self, view, scalar, prev_reward, prev_action, hp, cp, hg, cg):
        """仅取价值（bootstrap 估值用），不关心动作分布。"""
        with torch.no_grad():
            _, value, _, _, _, _, _ = self.forward(
                view, scalar, prev_reward, prev_action, hp, cp, hg, cg)
        return value

    def forward_seq(self, view, scalar, prev_reward, prev_action,
                    dones, hp, cp, hg, cg):
        """[P0-1] 序列前向：对 [T, B, ...] 的整段 rollout 重展开（BPTT）。

        这是 recurrent-PPO 的正确更新路径：从 rollout 起始隐状态出发，逐时间
        步前向，梯度贯穿全序列（对比旧版：把采集时存的状态当独立输入单步喂入，
        BPTT 长度=1，LSTM 学不到记忆）。逐步调用 forward，两种连接方式共用。

        done 掩码语义（与采集路径严格对齐）：采集时清零发生在 env.step 返回
        done 之后——即第 t 步 done=True 表示该步是 episode 最后一步，其输出
        状态被清掉，第 t+1 步从零状态开始。这里在消费第 t 步观测之前用
        dones[t-1] 掩状态（首步不掩），即可精确复现该语义。
        （GAE 评价第 t 个动作时读的是 dones[t]，时间位置不同，两处都对。）

        Args:
          view:        [T, B, 3, H, W] float（time-major，已归一化）。
          scalar:      [T, B, 6] float。
          prev_reward: [T, B] float（未裁剪，内部与 forward 同样 clamp）。
          prev_action: [T, B] long。
          dones:       [T, B] float/bool：第 t 步是否 episode 结束。
          hp,cp,hg,cg: [1, B, hidden] rollout 起始状态（可带梯度）。

        Returns:
          logits [T, B, A]、value [T, B, 1]、aux [T, B, 2] 或 None。
        """
        t_len = view.shape[0]
        logits_all, value_all, aux_all = [], [], []
        for t in range(t_len):
            if t > 0:
                # 第 t-1 步是 episode 终点 -> 该步之后的记忆清零。
                m = (1.0 - dones[t - 1].to(hp.dtype)).view(1, -1, 1)
                hp = hp * m
                cp = cp * m
                hg = hg * m
                cg = cg * m
            logits, value, hp, cp, hg, cg, aux = self.forward(
                view[t], scalar[t], prev_reward[t], prev_action[t],
                hp, cp, hg, cg)
            logits_all.append(logits)
            value_all.append(value)
            if aux is not None:
                aux_all.append(aux)
        logits = torch.stack(logits_all, dim=0)          # [T, B, A]
        value = torch.stack(value_all, dim=0)            # [T, B, 1]
        aux = torch.stack(aux_all, dim=0) if aux_all else None  # [T, B, 2]
        return logits, value, aux


def model_meta(agent):
    """PPOAgent -> 存进 checkpoint 的结构 / 遮蔽字段。"""
    return {key: getattr(agent, key) for key in MODEL_META_DEFAULTS}


def agent_kwargs_from_checkpoint(ckpt):
    """按 checkpoint 元数据给出 PPOAgent 的构造参数。

    训练续训比对、评估、轨迹可视化都用这一份规则。旧 checkpoint 没有结构 /
    遮蔽字段，按当时唯一的行为补齐：并联、不遮蔽。不带 pretrained_vision：
    权重随即被 checkpoint 覆盖，没必要再下载 ImageNet 预训练权重。
    """
    kwargs = {"num_actions": int(ckpt["num_actions"])}
    for key, default in MODEL_META_DEFAULTS.items():
        value = ckpt.get(key)
        kwargs[key] = default if value is None else type(default)(value)
    return kwargs


def count_parameters(agent):
    return sum(p.numel() for p in agent.parameters())


def describe_agent(agent):
    """一行说明网络结构与遮蔽设置（训练 / 评估 / 轨迹日志用）。"""
    masked = [name for name, on in (
        ("position", agent.mask_position),
        ("yaw", agent.mask_yaw),
        ("status_bar(%d rows)" % agent.mask_status_rows, agent.mask_status_rows > 0),
    ) if on]
    return ("topology=%s hidden=%d backbone=%s aux_heading=%s params=%s "
            "masked_inputs=%s" % (
                agent.model_topology, agent.hidden_size, agent.vision_backbone,
                agent.aux_heading, format(count_parameters(agent), ","),
                ",".join(masked) if masked else "none"))
