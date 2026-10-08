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

"""checkpoint 定位工具（不依赖 torch，run.py 与 ppo_train.py 共用）。

训练每次保存"最新状态"（周期性 ppo_iter_*.pt、每次课程测试后的 ppo_iter_*.pt、
ppo_final.pt）都会同时更新 checkpoint_latest.json，断点恢复时直接读它即可，
不必加载各个 .pt 比较 update。旧 run 没有这个文件时退回到编号最大的
ppo_iter_*.pt。
"""

from __future__ import annotations

import glob
import json
import os
import re
import time

LATEST_POINTER = "checkpoint_latest.json"


def iter_checkpoint_update(path):
    """ppo_iter_N.pt -> N；其他文件名返回 -1。"""
    match = re.search(r"ppo_iter_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def write_latest_pointer(run_dir, ckpt_path, update, global_step):
    """原子地记录"最新训练状态"所在的 checkpoint（文件名相对 run_dir）。"""
    pointer = os.path.join(run_dir, LATEST_POINTER)
    tmp = pointer + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"path": os.path.basename(ckpt_path), "update": int(update),
                   "global_step": int(global_step),
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    os.replace(tmp, pointer)


def find_latest_checkpoint(run_dir):
    """返回 run_dir 里最新训练状态的 checkpoint 路径，没有则返回 None。"""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    try:
        with open(os.path.join(run_dir, LATEST_POINTER), encoding="utf-8") as f:
            name = json.load(f).get("path")
        if name:
            path = os.path.join(run_dir, os.path.basename(name))
            if os.path.isfile(path):
                return path
    except (OSError, ValueError, AttributeError):
        pass
    iters = glob.glob(os.path.join(run_dir, "ppo_iter_*.pt"))
    if iters:
        return max(iters, key=iter_checkpoint_update)
    final = os.path.join(run_dir, "ppo_final.pt")
    return final if os.path.isfile(final) else None
