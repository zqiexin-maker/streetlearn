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

"""PyTorch PPO 策略/价值网络 —— CityNav 式（并联双 LSTM + 可选辅助 heading）。

============================================================================
P1 改造目标（对标 DeepMind StreetLearn 论文 CityNav，覆盖差异 2/3/9）
============================================================================
论文要点：
  - agent 只看第一人称 RGB 图像，从不看底层图。
  - 网络用「卷积编码器 → 多路 LSTM」：一个 policy LSTM + 一个 goal LSTM，
    两路 LSTM **并联**：都直接接收视觉特征，且都接收 r_{t-1} / a_{t-1}；
    两路 LSTM 输出 **concat** 后再接策略头与价值头。
  - Goal LSTM 额外接收目标描述符 g_t（论文默认绝对经纬度）；可在 Goal LSTM
    输出端加辅助 heading(θ) 预测任务。

本实现（P1 对齐论文并联结构）：
  观测模态（agent 不看底层图）：
    view_image  [B,3,H,W]  --[视觉编码器]--> view_feat (512)   第一人称街景
    scalar      [B,6]       = sin(yaw), cos(yaw), latlng(2), target_latlng(2)
    prev_reward [B,1]       上一步奖励 r_{t-1}（按 1/100 缩放，论文要求显式注入）
    prev_action [B,A]       上一步动作 a_{t-1} 的 one-hot（论文要求显式注入）

  网络（并联双 LSTM）：
    1) goal_lstm  ：输入 = cat(view_feat, goal_feat, state_feat, prev_reward, prev_action)
                     -> goal_hidden   （编码"目标+视觉+状态"的 goal 条件特征）
    2) policy_lstm：输入 = cat(view_feat, state_feat, prev_reward, prev_action)
                     -> policy_hidden （承载动作决策记忆）
    3) combined    = cat(goal_hidden, policy_hidden)         # 论文两路融合
    4) 双头：policy_head -> logits；value_head -> value
    5) 辅助 heading 头（--aux_heading 可开关）：从 goal_hidden 预测 (sinθ, cosθ)

  目标表示（差异 9）：
    --goal_type target_latlng（默认，论文最终选择） / target_landmarks（可选，
    当前未接入数据，启用即 NotImplementedError，满足"补充但不启用"）。

  视觉编码器（差异 5 预备）：
    --vision_backbone resnet18（默认，有意增强）/ small_cnn（论文 84×84 小卷积）。
============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 支持的目标表示；target_landmarks 仅占位、默认不启用（差异 9）。
SUPPORTED_GOAL_TYPES = ("target_latlng", "target_landmarks")

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

    论文靠第一人称图像隐式携带朝向/位置；我们显式给出以稳定训练，
    保留此前 6 维标量里"当前位姿"这部分有效信号（差异 3 之外额外保留的信息）。
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
    """循环（recurrent）策略/价值网络，CityNav 式**并联**双 LSTM 结构。"""

    def __init__(
        self,
        num_actions: int,
        hidden_size: int = 512,
        aux_heading: bool = False,
        pretrained_vision: bool = False,
        vision_backbone: str = "resnet18",
        goal_type: str = "target_latlng",
    ):
        super().__init__()
        if goal_type not in SUPPORTED_GOAL_TYPES:
            raise ValueError(
                "goal_type must be one of %s, got %r" % (SUPPORTED_GOAL_TYPES, goal_type))
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.aux_heading = aux_heading
        self.goal_type = goal_type
        self.vision_backbone = vision_backbone

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

        # 论文并联结构：goal / policy 两路 LSTM 都直接接 view_feat，
        # 并且都接收 prev_reward(1) 与 prev_action one-hot(num_actions)。
        goal_in = vd + gd + sd + 1 + num_actions
        policy_in = vd + sd + 1 + num_actions
        self.goal_lstm = nn.LSTM(goal_in, hidden_size, num_layers=1,
                                 batch_first=False)
        self.policy_lstm = nn.LSTM(policy_in, hidden_size, num_layers=1,
                                   batch_first=False)

        # 双头：从两路 LSTM 输出 concat(2*hidden) 得到。
        self.policy_head = nn.Linear(2 * hidden_size, num_actions)
        self.value_head = nn.Linear(2 * hidden_size, 1)

        # 辅助 heading 头（可选）：从 goal_hidden 预测 (sinθ, cosθ)。
        if aux_heading:
            self.heading_head = nn.Linear(hidden_size, 2)

    def initial_state(self, batch_size: int = 1, device=None):
        """返回全零的双 LSTM 初始状态 (hp, cp, hg, cg)，形状均为 [1, B, hidden]。"""
        z = torch.zeros(1, batch_size, self.hidden_size,
                        dtype=torch.float32, device=device)
        return z, z.clone(), z.clone(), z.clone()  # (hp, cp, hg, cg)

    def _encode(self, view, scalar):
        """视觉 + 目标 + 状态 三路编码。

        返回 (view_f [B,512], goal_f [B,64], state_f [B,64])。
        scalar 固定布局：[0:2]=yaw sin/cos, [2:4]=当前 latlng, [4:6]=目标 latlng。
        """
        view_f = self.vision(view)                    # [B, 512]
        goal_f = self.goal_embed(scalar[:, 4:6])      # [B, 64] 目标经纬度
        state_f = self.state_embed(scalar[:, 0:4])    # [B, 64] 当前位姿
        return view_f, goal_f, state_f

    def forward(self, view, scalar, prev_reward, prev_action, hp, cp, hg, cg):
        """单步前向（seq_len=1）。

        Args:
          view:        [B, 3, H, W] float，已做 ImageNet 归一化。
          scalar:      [B, 6] float（yaw sin/cos + latlng + target_latlng）。
          prev_reward: [B] 或 [B,1] float，上一步奖励 r_{t-1}（原值，内部 clamp 到 [-1,1]）。
          prev_action: [B] long，上一步动作 a_{t-1}（用于 one-hot）。
          hp,cp: policy_lstm 隐/细胞状态 [1, B, hidden]。
          hg,cg: goal_lstm  隐/细胞状态 [1, B, hidden]。

        Returns（调用者必须按此顺序解包）:
          logits: [B, num_actions]   策略头原始输出
          value:  [B, 1]             价值头状态价值估计
          hp,cp:  更新后的 policy_lstm 状态
          hg,cg:  更新后的 goal_lstm 状态
          aux:    [B, 2] 预测 (sinθ,cosθ)；若关闭 aux_heading 则为 None
        """
        view_f, goal_f, state_f = self._encode(view, scalar)
        # [P0-2] 论文语义：进入网络的上一步奖励裁剪到 [-1,1]（与 GAE 侧统一，
        # 替换旧的 /REWARD_SCALE —— 两处不一致曾是价值损失爆量的根因之一）。
        pr = prev_reward.view(-1, 1).to(view_f.dtype).clamp_(-1.0, 1.0)  # [B, 1]
        pa = F.one_hot(prev_action.long().view(-1),
                       self.num_actions).to(view_f.dtype)  # [B, A]

        # ---- goal LSTM（并联第一路）----
        g_in = torch.cat([view_f, goal_f, state_f, pr, pa], dim=1).unsqueeze(0)
        gh, (hg, cg) = self.goal_lstm(g_in, (hg, cg))
        gh = gh.squeeze(0)                                 # [B, hidden]
        # ---- policy LSTM（并联第二路）----
        p_in = torch.cat([view_f, state_f, pr, pa], dim=1).unsqueeze(0)
        ph, (hp, cp) = self.policy_lstm(p_in, (hp, cp))
        ph = ph.squeeze(0)                                 # [B, hidden]
        # ---- 两路 concat 给双头 ----
        combined = torch.cat([gh, ph], dim=1)              # [B, 2*hidden]
        logits = self.policy_head(combined)                # [B, num_actions]
        value = self.value_head(combined)                  # [B, 1]
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
        BPTT 长度=1，LSTM 学不到记忆）。

        done 掩码语义（与采集路径严格对齐）：采集时清零发生在 env.step 返回
        done 之后——即第 t 步 done=True 表示该步是 episode 最后一步，其输出
        状态被清掉，第 t+1 步从零状态开始。这里在消费第 t 步观测之前用
        dones[t-1] 掩状态（首步不掩），即可精确复现该语义。

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
