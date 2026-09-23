# Copyright 2018 Google LLC.
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""StreetLearn level for the courier task with random goals/targets.

In this environment, the agent receives a reward for every coin it collects and
an extra reward for locating the goal pano.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import random
import six

from streetlearn.engine.python import color
from streetlearn.python.environment import coin_game


class CourierGame(coin_game.CoinGame):
  """Coin game that gives extra reward for finding the goal pano. A courier goal
  is randomly selected from panos in the graph. On success or timeout, a new
  goal is chosen. The episode ends after a fixed episode length.
  """

  def __init__(self, config):
    """Constructor.

    This coin game gives extra reward for finding the goal pano, and resets the
    goal once the goal has been found (or on timeout). Panos can be assigned
    rewards (coins) randomly and the agent will receive the reward the first
    time they visit these panos.

    Args:
      config: config dict of various settings.
    """
    super(CourierGame, self).__init__(config)
    self._reward_current_goal = config['max_reward_per_goal']
    self._min_radius_meters = config['min_radius_meters']
    self._max_radius_meters = config['max_radius_meters']
    self._goal_timeout = config['goal_timeout']
    self._colors['goal'] = color.Color(*config['color_for_goal'])
    self._colors['shortest_path'] = color.Color(
        *config['color_for_shortest_path'])

    self._num_steps_this_goal = 0
    self._success_inverse_path_len = []
    self._min_distance_reached = np.finfo(np.float32).max
    self._initial_distance_to_goal = np.finfo(np.float32).max
    self._current_goal_id = None
    self._visited_panos = set()
    self._shortest_path = {}
    self._timed_out = False

    # ---- P1/P2 扩展状态 ----
    # 课程学习：目标与当前位置的最大距离（米）。None = 不限制（全图）。
    self._max_target_distance = None
    # held-out 泛化：目标只能从这个集合里抽（None = 不限制）。
    self._allowed_goal_set = None
    # held-out 拆分参数（延迟到首次选目标时按图计算）。
    self._held_out_fraction = 0.0
    self._held_out_seed = 0
    self._use_held_out = True
    self._held_out_cache = None
    # 本 episode 累计的"到达全奖"（论文口径，排除塑形/硬币）与 BFS Oracle 上界。
    self._episode_arrival_reward = 0.0
    self._episode_oracle_reward = 0.0

    # ---- 评估 head-to-head 扩展 ----
    # 目标采样独立 RNG（由 --goal_seed 决定），与全局 np.random 解耦，保证
    # Agent/Oracle 两遍评估抽到完全一致的目标序列。None=沿用全局 np.random（训练/旧行为）。
    self._goal_rng = None
    # 单目标/episode 模式：每 episode 只分配 1 个目标，到达后不再重抽，
    # 使目标抽取次数 == episode 数，两边严格同步，实现真正的 head-to-head。
    self._single_goal = False
    # 本 episode 是否已到达目标（单目标模式下去重计分用）。
    self._reached_this_episode = False

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Selects a random pano as goal destination and resets episode statistics.

    If there are any coins, clears the set of touched panos and randomly
    generates reward-yielding coins and populates pano_id_to_color.

    Args:
      streetlearn: a streetlearn instance.
    Returns:
      A newly populated pano_id_to_color dictionary.
    """
    # Populate the list of panos and assign optional coins to panos.
    # ① 父类CoinGame初始化（设置硬币节点）
    pano_id_to_color = super(CourierGame, self).on_reset(streetlearn)

    # 重置本 episode 的累计奖励统计（必须在选目标之前，否则会清掉
    # 首个目标在 _pick_random_goal 里累加的 oracle 计数）。
    self._episode_arrival_reward = 0.0
    self._episode_oracle_reward = 0.0
    self._reached_this_episode = False

    # Assign the goal location to one of the panos.
    # ② ★ 随机选择目标并计算最短路径
    self._pick_random_goal(streetlearn)

    # Resets the episode statistics.
    # ③ 重置统计
    self._num_steps_this_goal = 0
    self._success_inverse_path_len = []
    return pano_id_to_color

  @property
  def goal_id(self):
    """Returns the ID of the goal Pano."""
    return self._current_goal_id

  def get_reward(self, streetlearn):
    """Looks at current_pano_id and collects any reward found there.

    Args:
      streetlearn: A streetlearn instance.
    Returns:
      reward: the reward from the last step.
    """
    # If we have exceeded the maximum steps to look for a goal, reset the goal.
    # ① 超时检查
    if self._num_steps_this_goal > self._goal_timeout:
      logging.info('seed %d, frame %d: courier target TIMEOUT (%d steps)',
                   streetlearn.seed, streetlearn.frame_count,
                   self._num_steps_this_goal)
      self._num_steps_this_goal = 0
      if not self._single_goal:
        self._pick_random_goal(streetlearn)  # 重新选目标（多目标模式）
      # 单目标模式：保持同一目标，继续尝试直到 episode 结束。

    # ② ★ 计算奖励
    (reward, found_goal) = self._compute_reward(streetlearn)  # → 跳转_compute_reward方法

    # If we have found the goal, set a new one and update episode statistics.
    # ③ 成功→重选目标 + 记录SPL
    if found_goal:
      logging.info('seed %d, frame %d: courier target FOUND (%d steps)',
                   streetlearn.seed, streetlearn.frame_count,
                   self._num_steps_this_goal)
      # P2a：累计本 episode 的"到达全奖"（论文口径，排除塑形/硬币）。
      self._episode_arrival_reward += self._reward_current_goal
      _, num_remaining_panos_to_goal_center = self._shortest_paths(
        streetlearn, self._current_goal_id, streetlearn.current_pano_id)
      self._success_inverse_path_len.append(
          self._compute_spl_current_goal(streetlearn))
      if self._single_goal:
        # 单目标 head-to-head：标记已到达，不再重抽（避免重复计分 / 破坏目标对齐）。
        self._reached_this_episode = True
      else:
        self._pick_random_goal(streetlearn)
        self._num_steps_this_goal = 0

    # Give additional reward if current pano has a coin.
    # ④ 硬币奖励叠加
    if streetlearn.current_pano_id in self._coin_pano_id_set:
      reward += self._reward_per_coin
      self._coin_pano_id_set.remove(streetlearn.current_pano_id)

    self._num_steps_this_goal += 1
    return reward

  def get_info(self, streetlearn):
    """"Returns current information about the state of the environment.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      info: information from the environment at the last step.
    """
    info = super(CourierGame, self).get_info(streetlearn)
    info['num_steps_this_goal'] = self._num_steps_this_goal
    info['current_goal_id'] = self._current_goal_id
    info['min_distance_reached'] = self._min_distance_reached
    info['initial_distance_to_goal'] = self._initial_distance_to_goal
    info['reward_current_goal'] = self._reward_current_goal
    num_successes = len(self._success_inverse_path_len)
    info['num_successes'] = num_successes
    info['spl'] = sum(self._success_inverse_path_len) / (num_successes + 1)
    if num_successes > 0:
      info['spl_without_last_goal'] = (
          sum(self._success_inverse_path_len) / num_successes)
    else:
      info['spl_without_last_goal'] = 0
    # 用 .get() 容错（避免 current_pano 不在路径表中抛 KeyError）；若 next 为
    # None 或已等于当前 pano（即已到达目标），bearing 无意义，置 0 防止
    # GetPanoBearing(None/goal,...) 返回 None 后 None-float 减法崩溃。
    # 单目标 head-to-head 模式下，到达目标后当 episode 剩余步仍会调用 get_info，
    # 此时 current_pano==goal，_panos_to_goal 映射为 None，必须走此分支。
    next_pano_id = self._panos_to_goal.get(streetlearn.current_pano_id)
    info['next_pano_id'] = next_pano_id
    if next_pano_id is None or next_pano_id == streetlearn.current_pano_id:
      info['bearing_to_next_pano'] = 0.0
    else:
      b = streetlearn.engine.GetPanoBearing(
          streetlearn.current_pano_id, next_pano_id)
      if b is None:
        # 极少数情况下引擎对合法 pano 对返回 None（元数据缺失等），兜底置 0。
        info['bearing_to_next_pano'] = 0.0
      else:
        bearing_to_next_pano = b - streetlearn.engine.GetYaw()
        info['bearing_to_next_pano'] = (bearing_to_next_pano + 180) % 360 - 180
    # P2a：暴露本 episode 的"到达全奖"（论文口径）与 BFS Oracle 上界，供训练
    # /评估循环统计与论文 Table 2 对齐的指标。
    info['arrival_reward'] = self._episode_arrival_reward
    info['oracle_reward'] = self._episode_oracle_reward
    info['coverage'] = (self._episode_arrival_reward /
                         self._episode_oracle_reward
                         if self._episode_oracle_reward > 0 else 0.0)
    return info

  def done(self):
    """Returns a flag indicating the end of the current episode.

    This game ends only at the end of the episode or if the goal times out.
    During a single episode, every time a goal is found, a new one is chosen,
    until the time runs out.
    """
    if self._timed_out:
      self._timed_out = False
      return True
    else:
      return False

  def _sample_random_goal(self, streetlearn):
    """Randomly sets a new pano for the current goal.

    受两类约束（P1 课程学习 / held-out 泛化）：
      - 目标白名单：held-out 评估时只允许从 allowed 集合抽（None=不限制）。
      - 课程学习：目标与当前位置的距离不超过 _max_target_distance（None=不限制）。

    Args:
      streetlearn: The StreetLearn environment.
    """
    # 有效目标白名单（held-out 泛化评估用；None 表示不限制）。
    allowed = self._allowed_goal_set
    if allowed is None and self._held_out_fraction > 0:
      allowed = self._ensure_held_out_mask(streetlearn)

    # 基础候选：图中所有 pano，排除当前位置与上一个目标。
    base = [goal for goal in streetlearn.graph
            if ((goal != self._current_goal_id) and
                (goal != streetlearn.current_pano_id))]
    pool = base
    if allowed is not None:
      pool = [goal for goal in pool if goal in allowed]
    # 课程学习：限制目标与当前位置的距离不超过 _max_target_distance。
    if self._max_target_distance is not None:
      cur = streetlearn.current_pano_id
      maxd = self._max_target_distance
      dpool = [goal for goal in pool
               if streetlearn.engine.GetPanoDistance(cur, goal) <= maxd]
      if dpool:
        pool = dpool   # 仅当距离约束下仍有候选才采用，避免约束过紧导致无候选
    if not pool:
      pool = base      # 极端兜底：放宽所有约束

    # goal_seed 模式：用独立的、由 goal_seed 决定的 RNG，保证 Agent/Oracle 两遍
    # 评估抽到完全相同的目标序列（再配合单目标模式，实现严格 head-to-head）。
    # 未设置 goal_seed 时回退到全局 np.random（训练 / 旧行为，非确定性）。
    rng = self._goal_rng if self._goal_rng is not None else np.random
    self._current_goal_id = rng.choice(pool)
    self._min_distance_reached = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, self._current_goal_id)
    self._initial_distance_to_goal = self._min_distance_reached

  # ---------------------------------------------------------------------------
  # P1/P2 控制接口（供训练循环 / 评估脚本通过 SubprocVecEnv 调用）
  # ---------------------------------------------------------------------------
  def set_max_target_distance(self, meters):
    """课程学习：限制新目标与当前位置的最大距离（米）。meters=None 解除限制。"""
    self._max_target_distance = meters

  def set_allowed_goal_set(self, pano_set):
    """泛化评估：限制目标只能从给定 pano 集合抽（None=不限制）。"""
    self._allowed_goal_set = pano_set

  def set_goal_seed(self, seed):
    """评估 head-to-head：用独立 RNG 固定目标采样，使 Agent/Oracle 两遍评估
    抽到完全相同的目标序列。seed 通常取 goal_seed + worker_id 偏移，保证各
    worker 目标不同但跨 run 可复现。未调用则沿用全局 np.random（非确定性）。"""
    self._goal_rng = np.random.RandomState(seed)

  def set_single_goal_mode(self, flag):
    """评估 head-to-head：每 episode 只分配 1 个目标（到达后不重抽），
    使目标抽取次数 == episode 数，从而 Agent/Oracle 两遍面对完全一致的
    (起点, 目标) 序列，得到严格的论文口径 head-to-head 对比。"""
    self._single_goal = bool(flag)

  def setup_held_out(self, fraction, seed=0, use_held_out=True):
    """配置 held-out 泛化评估：把 pano 集合按确定性哈希拆成 in/out 两部分。

    Args:
      fraction: 被留出(out)的比例（论文用 0.25）。
      seed: 拆分随机种子（独立于训练 RNG，保证可复现）。
      use_held_out: True -> 目标只从 out 集合（未训练见过的）抽；
                    False -> 目标只从 in 集合（训练见过的）抽。
    实际拆分延迟到首次选目标时按图计算（见 _ensure_held_out_mask）。
    """
    self._held_out_fraction = fraction
    self._held_out_seed = seed
    self._use_held_out = use_held_out
    self._held_out_cache = None

  def _ensure_held_out_mask(self, streetlearn):
    """按图确定性地把 pano 拆成 in/out 两部分，返回生效的目标白名单。

    使用 Python random（与训练 RNG 隔离），保证拆分可复现且跨进程一致。
    """
    if self._held_out_cache is not None:
      return self._held_out_cache
    keys = sorted(streetlearn.graph.keys())
    rng = random.Random(self._held_out_seed)
    order = list(keys)
    rng.shuffle(order)
    k = int(round(len(order) * self._held_out_fraction))
    out_set = set(order[:k])
    in_set = set(order[k:])
    self._held_out_cache = out_set if self._use_held_out else in_set
    return self._held_out_cache

  def _pick_random_goal(self, streetlearn):
    """Randomly sets a new pano for the current goal.

    Args:
      streetlearn: The StreetLearn environment.
    """
    self._visited_panos.clear()
    self._sample_random_goal(streetlearn)
    logging.info('seed %d, frame %d: new goal id: %s distance: %f',
                 streetlearn.seed, streetlearn.frame_count, self.goal_id,
                 self._min_distance_reached)
    pano_data = streetlearn.engine.GetMetadata(self.goal_id).pano
    logging.info('seed %d: new goal at (%f, %f)',
                 streetlearn.seed, pano_data.coords.lat, pano_data.coords.lng)

    # Compute the extended graph and shortest path to goal to estimate
    # the reward to give to the agent.
    shortest_path, num_panos = self._shortest_paths(
        streetlearn, self._current_goal_id, streetlearn.current_pano_id)
    self._reward_current_goal = num_panos
    # P2b：BFS Oracle 上界——每个被指派的目标，其"最优可达奖励"= 最短路径节点数。
    # 一整个 episode 累加，即为该 episode 的理论满分（agent 到达全部分派目标时）。
    self._episode_oracle_reward += num_panos
    logging.info('seed %d: goal reward depends on #panos to goal: %d',
                 streetlearn.seed, self._reward_current_goal)
    # Decorate the graph.
    self._pano_id_to_color = {coin_pano_id: self._colors['coin']
                              for coin_pano_id in self._coin_pano_id_set}
    self._update_pano_id_to_color()
    for pano_id in six.iterkeys(shortest_path):
      self._pano_id_to_color[pano_id] = self._colors['shortest_path']
    self._pano_id_to_color[self.goal_id] = self._colors['goal']

  def _compute_reward(self, streetlearn):
    """Reward is a piecewise linear function of distance to the goal.

    If agent is greater than max_radius_meters from the goal, reward is 0. If
    agent is less than min_radius_reters, reward is max_reward_per_goal. Between
    min and max radius the reward is linear between 0 and max_reward_per_goal.

    Args:
      streetlearn: The StreetLearn environment.
    Returns:
      The reward for the current step.
    奖励设计的三层结构：

    到达奖励（< min_radius）= 最短路径长度 → SPL指标的基础
    塑形奖励（min_radius ~ max_radius）= 线性衰减，只在距离递减时才给 → 防止来回走刷分
    硬币奖励（独立叠加）= 稠密小奖励，鼓励探索
    """
    # Do not give rewards for already visited panos.
    reward = 0
    found_goal = False
    # 单目标 head-to-head：本 episode 已到达过目标则不再给奖励 / 不再判定到达，
    # 避免同一目标被反复计分时虚高 arrival_reward。
    if self._single_goal and self._reached_this_episode:
      return (reward, found_goal)
    # 已访问节点不给奖励（防止原地转圈刷分）
    if streetlearn.current_pano_id in self._visited_panos:
      return (reward, found_goal)

    # Mark the pano as visited and compute distance to goal.
    self._visited_panos.add(streetlearn.current_pano_id)
    distance_to_goal = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, self.goal_id)
    if distance_to_goal < self._min_radius_meters:
      # Have we reached the goal?
      # ★ 到达目标！奖励 = 最短路径节点数（近似路径长度）
      reward = self._reward_current_goal
      found_goal = True
      logging.info(
          'seed %d, frame %d: reached goal, distance_to_goal=%s, reward=%s',
          streetlearn.seed, streetlearn.frame_count, distance_to_goal, reward)
    else:
      if distance_to_goal < self._max_radius_meters:
        # Early reward shaping.
        # ★ 中间区域：分段线性塑形奖励
        # 只有当距离比之前最近距离更近时才给奖励（单调递减才奖）
        if distance_to_goal < self._min_distance_reached:
          reward = (self._reward_current_goal *
                    (self._max_radius_meters - distance_to_goal) /
                    (self._max_radius_meters - self._min_radius_meters))
          self._min_distance_reached = distance_to_goal
          logging.info('seed %d, frame %d: distance_to_goal=%f, reward=%d',
                       streetlearn.seed, streetlearn.frame_count,
                       distance_to_goal, reward)

    return (reward, found_goal)

  def _compute_spl_current_goal(self, streetlearn):
    """Compute the success weighted by inverse path length for the current goal.

    We use the SPL definition from Eq. 1 in the following paper:
    Anderson et al. (2018) "On Evaluation of Embodied Navigation Agents"
    https://arxiv.org/pdf/1807.06757.pdf

    Args:
      streetlearn: The StreetLearn environment.
    Returns:
      The SPL metric for the current goal.
    """
    # Since reaching the goal is defined as being within a circle around the
    # goal pano, we subtract the panoramas within that circle from the shortest
    # path length estimate, as well as from the actual path length.
    # We add 1 to handle cases when the agent spawned within that circle.
    _, num_remaining_panos_to_goal = self._shortest_paths(
        streetlearn, self._current_goal_id, streetlearn.current_pano_id)
    shortest_path_len = self._reward_current_goal - num_remaining_panos_to_goal
    shortest_path_len = max(shortest_path_len, 1)
    actual_path_len = len(self._visited_panos) - num_remaining_panos_to_goal
    actual_path_len = max(actual_path_len, 1)
    return shortest_path_len / max(actual_path_len, shortest_path_len)

