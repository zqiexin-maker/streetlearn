# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agents package for StreetLearn (PyTorch PPO).

  - policy network : streetlearn/python/agents/ppo_agent.py (PPOAgent)
  - training loop  : streetlearn/python/ppo_train.py
  - evaluation     : streetlearn/python/ppo_eval.py

The original IMPALA / TensorFlow-Sonnet agents (CityNavAgent, GoalNavAgent,
PlainAgent, LocalePathway) and their experiment.py training entry have been
removed; the PyTorch PPO stack replaces them.
"""
