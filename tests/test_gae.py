"""compute_gae 的数值检查（修改意见 1：GAE 结束标志错位）。

dones[:, t] = 执行第 t 个动作之后 episode 是否结束（与 ppo_train 里 done_store 一致）。
运行：python run.py selftest，或 python -m unittest tests.test_gae -v
"""
import unittest

import numpy as np

from streetlearn.python.ppo_train import GAE_VERSION, compute_gae, compute_gae_v1

GAMMA, LAM = 0.99, 0.95


def reference_gae(rewards, values, dones, last_values, gamma=GAMMA, lam=LAM):
    """按定义正向求和：A_t = sum_l (gamma*lam)^l * delta_{t+l}，遇到本 episode
    的终止步（含）就停；delta 在终止步不接下一状态的价值。与 compute_gae 的
    逆序递推写法无关，用来交叉检查。"""
    n, t_len = rewards.shape
    adv = np.zeros((n, t_len), dtype=np.float64)
    for e in range(n):
        for t in range(t_len):
            total, discount = 0.0, 1.0
            for k in range(t, t_len):
                next_v = values[e, k + 1] if k + 1 < t_len else last_values[e]
                delta = (rewards[e, k] + gamma * next_v * (1.0 - dones[e, k])
                         - values[e, k])
                total += discount * delta
                if dones[e, k]:
                    break
                discount *= gamma * lam
            adv[e, t] = total
    return adv


def gae(rewards, values, dones, last_values):
    return compute_gae(np.asarray(rewards, dtype=np.float32),
                       np.asarray(values, dtype=np.float32),
                       np.asarray(dones, dtype=bool),
                       np.asarray(last_values, dtype=np.float32), GAMMA, LAM)


class ComputeGaeTest(unittest.TestCase):

    def test_document_example(self):
        # 修改意见里的三步例子：旧版得到 [0, 99.0595, -1]，正确为 [0.9405, 1, -1]。
        adv, ret = gae([[0, 1, 0]], [[0, 0, 100]], [[False, True, False]], [100])
        np.testing.assert_allclose(adv[0], [0.9405, 1.0, -1.0], atol=1e-5)
        np.testing.assert_allclose(ret, adv + np.array([[0, 0, 100]]), atol=1e-5)

    def test_no_episode_end(self):
        rng = np.random.RandomState(0)
        rewards, values = rng.randn(3, 6), rng.randn(3, 6)
        dones = np.zeros((3, 6), dtype=bool)
        last = rng.randn(3)
        adv, _ = gae(rewards, values, dones, last)
        np.testing.assert_allclose(adv, reference_gae(rewards, values, dones, last),
                                   atol=1e-4)

    def test_end_inside_segment(self):
        rewards = [[0.0, 0.5, 1.0, 0.0, 0.2]]
        values = [[0.3, 0.1, 0.4, 100.0, 7.0]]
        dones = [[False, False, True, False, False]]
        adv, _ = gae(rewards, values, dones, [5.0])
        np.testing.assert_allclose(
            adv, reference_gae(np.array(rewards), np.array(values),
                               np.array(dones), np.array([5.0])), atol=1e-4)
        # 终止后新 episode 的价值从 100 改成 1000（连同 bootstrap 值），
        # 旧 episode 终止步及之前的优势都不能变。
        changed, _ = gae(rewards, [[0.3, 0.1, 0.4, 1000.0, -3.0]], dones, [-50.0])
        np.testing.assert_allclose(changed[0, :3], adv[0, :3], atol=1e-6)
        # 终止步只剩自己的奖励：A = r - V。
        self.assertAlmostEqual(float(adv[0, 2]), 1.0 - 0.4, places=5)

    def test_end_at_last_step(self):
        rewards = [[0.0, 0.0, 1.0]]
        values = [[0.2, 0.5, 0.9]]
        dones = [[False, False, True]]
        adv_a, _ = gae(rewards, values, dones, [0.0])
        adv_b, _ = gae(rewards, values, dones, [1000.0])
        # 最后一步已结束：rollout 之后的 bootstrap 价值不应影响任何一步。
        np.testing.assert_allclose(adv_a, adv_b, atol=1e-6)
        np.testing.assert_allclose(
            adv_a, reference_gae(np.array(rewards), np.array(values),
                                 np.array(dones), np.array([0.0])), atol=1e-5)

    def test_consecutive_one_step_episodes(self):
        rewards = [[1.0, -0.5, 0.25]]
        values = [[0.4, 0.3, 0.8]]
        dones = [[True, True, True]]
        adv, ret = gae(rewards, values, dones, [123.0])
        # 每步都是一个完整 episode：优势 = r - V，回报 = r。
        np.testing.assert_allclose(adv[0], [0.6, -0.8, -0.55], atol=1e-6)
        np.testing.assert_allclose(ret[0], rewards[0], atol=1e-6)

    def test_random_batch_matches_definition(self):
        rng = np.random.RandomState(1)
        rewards = rng.randn(8, 32)
        values = rng.randn(8, 32) * 3.0
        dones = rng.rand(8, 32) < 0.15
        last = rng.randn(8)
        adv, ret = gae(rewards, values, dones, last)
        np.testing.assert_allclose(adv, reference_gae(rewards, values, dones, last),
                                   atol=1e-4)
        np.testing.assert_allclose(ret, adv + values, atol=1e-4)

    def test_checkpoint_version_marks_fix(self):
        self.assertEqual(GAE_VERSION, 2)

    def test_legacy_version_reproduces_old_numbers(self):
        # --gae_version 1（对照实验用）必须原样复现修复前的结果。
        adv, _ = compute_gae_v1(np.array([[0, 1, 0]], dtype=np.float32),
                                np.array([[0, 0, 100]], dtype=np.float32),
                                np.array([[False, True, False]]),
                                np.array([100], dtype=np.float32), GAMMA, LAM)
        np.testing.assert_allclose(adv[0], [0.0, 99.0595, -1.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
