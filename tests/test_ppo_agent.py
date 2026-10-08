"""PPOAgent 的结构与输入遮蔽检查（修改意见 3、4）。

只用 CPU 和小网络（small_cnn、hidden=32），十几秒内跑完。
运行：python run.py selftest，或 python -m unittest tests.test_ppo_agent -v
"""
import importlib.util
import io
import unittest

import numpy as np
import torch

from streetlearn.python.agents.ppo_agent import (
    GOAL_MESSAGE_DIM,
    MODEL_META_DEFAULTS,
    PPOAgent,
    SUPPORTED_TOPOLOGIES,
    agent_kwargs_from_checkpoint,
    describe_agent,
    model_meta,
)
from streetlearn.python.ppo_train import obs_to_batch

A, B, HIDDEN = 5, 3, 32
H = W = 84
STATUS_ROWS = 10


def make_agent(topology="parallel", **kwargs):
    torch.manual_seed(0)
    return PPOAgent(num_actions=A, hidden_size=HIDDEN, vision_backbone="small_cnn",
                    model_topology=topology, **kwargs)


def make_inputs(batch=B, seed=0):
    g = torch.Generator().manual_seed(seed)
    view = torch.randn(batch, 3, H, W, generator=g)
    scalar = torch.rand(batch, 6, generator=g)
    prev_r = torch.randn(batch, generator=g)
    prev_a = torch.randint(0, A, (batch,), generator=g)
    return view, scalar, prev_r, prev_a


def step(agent, view, scalar, prev_r, prev_a, state=None):
    state = state or agent.initial_state(view.shape[0])
    with torch.no_grad():
        return agent(view, scalar, prev_r, prev_a, *state)


class TopologyTest(unittest.TestCase):

    def test_single_step_shapes(self):
        for topology in SUPPORTED_TOPOLOGIES:
            with self.subTest(topology=topology):
                agent = make_agent(topology, aux_heading=True)
                logits, value, hp, cp, hg, cg, aux = step(agent, *make_inputs())
                self.assertEqual(tuple(logits.shape), (B, A))
                self.assertEqual(tuple(value.shape), (B, 1))
                self.assertEqual(tuple(aux.shape), (B, 2))
                for s in (hp, cp, hg, cg):
                    self.assertEqual(tuple(s.shape), (1, B, HIDDEN))

    def test_resnet18_dimensions(self):
        # ResNet18 下 policy LSTM 输入 582 -> 646 维，双头输入 1024 -> 512 维。
        if importlib.util.find_spec("torchvision") is None:
            self.skipTest("torchvision not installed")
        parallel = PPOAgent(num_actions=A, model_topology="parallel")
        serial = PPOAgent(num_actions=A, model_topology="serial")
        self.assertEqual(parallel.policy_lstm.input_size, 582)
        self.assertEqual(serial.policy_lstm.input_size, 582 + GOAL_MESSAGE_DIM)
        self.assertEqual(parallel.policy_head.in_features, 1024)
        self.assertEqual(serial.policy_head.in_features, 512)
        self.assertEqual(serial.value_head.in_features, 512)
        self.assertEqual(serial.goal_to_policy[0].out_features, GOAL_MESSAGE_DIM)
        self.assertFalse(hasattr(parallel, "goal_to_policy"))
        # 两路 LSTM 的隐状态仍是 [1, B, 512]。
        self.assertEqual(serial.goal_lstm.hidden_size, 512)
        self.assertEqual(serial.policy_lstm.hidden_size, 512)

    def test_sequence_forward_and_gradients(self):
        t_len = 4
        for topology in SUPPORTED_TOPOLOGIES:
            with self.subTest(topology=topology):
                agent = make_agent(topology)
                g = torch.Generator().manual_seed(1)
                view = torch.randn(t_len, B, 3, H, W, generator=g)
                scalar = torch.rand(t_len, B, 6, generator=g)
                prev_r = torch.randn(t_len, B, generator=g)
                prev_a = torch.randint(0, A, (t_len, B), generator=g)
                dones = torch.zeros(t_len, B)
                logits, value, aux = agent(view, scalar, prev_r, prev_a,
                                           *agent.initial_state(B), dones=dones)
                self.assertEqual(tuple(logits.shape), (t_len, B, A))
                self.assertEqual(tuple(value.shape), (t_len, B, 1))
                self.assertIsNone(aux)
                # 只对价值反传：串联时 goal LSTM 只能经瓶颈和 policy LSTM 拿到梯度。
                value.sum().backward()
                modules = [agent.goal_lstm, agent.policy_lstm]
                if topology == "serial":
                    modules.append(agent.goal_to_policy)
                for module in modules:
                    for name, param in module.named_parameters():
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                    self.assertGreater(
                        sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_serial_policy_branch_reads_goal(self):
        # 串联时双头只读 policy LSTM；改目标经纬度能改变输出，说明目标信息经
        # goal LSTM 和瓶颈进入了 policy LSTM。
        agent = make_agent("serial")
        view, scalar, prev_r, prev_a = make_inputs()
        moved = scalar.clone()
        moved[:, 4:6] = 1.0 - moved[:, 4:6]
        logits_a = step(agent, view, scalar, prev_r, prev_a)[0]
        logits_b = step(agent, view, moved, prev_r, prev_a)[0]
        self.assertGreater((logits_a - logits_b).abs().max().item(), 1e-6)

    def test_done_inside_sequence_clears_memory(self):
        t_len = 3
        for topology in SUPPORTED_TOPOLOGIES:
            with self.subTest(topology=topology):
                agent = make_agent(topology).eval()
                g = torch.Generator().manual_seed(2)
                view = torch.randn(t_len, B, 3, H, W, generator=g)
                scalar = torch.rand(t_len, B, 6, generator=g)
                prev_r = torch.zeros(t_len, B)
                prev_a = torch.zeros(t_len, B, dtype=torch.long)
                dones = torch.tensor([[0.0] * B, [1.0] * B, [0.0] * B])
                with torch.no_grad():
                    logits, value, _ = agent(view, scalar, prev_r, prev_a,
                                             *agent.initial_state(B), dones=dones)
                # 第 1 步结束了 episode，第 2 步应从零记忆开始。
                fresh = step(agent, view[2], scalar[2], prev_r[2], prev_a[2])
                torch.testing.assert_close(logits[2], fresh[0])
                torch.testing.assert_close(value[2], fresh[1])


class CheckpointTest(unittest.TestCase):

    def roundtrip(self, agent):
        buffer = io.BytesIO()
        torch.save({"agent": agent.state_dict(), "num_actions": A,
                    **model_meta(agent)}, buffer)
        buffer.seek(0)
        ckpt = torch.load(buffer, map_location="cpu", weights_only=False)
        restored = PPOAgent(**agent_kwargs_from_checkpoint(ckpt))
        restored.load_state_dict(ckpt["agent"])
        return restored.eval()

    def test_save_and_reload_gives_same_outputs(self):
        for topology in SUPPORTED_TOPOLOGIES:
            with self.subTest(topology=topology):
                agent = make_agent(topology, mask_yaw=True,
                                   mask_status_rows=STATUS_ROWS).eval()
                restored = self.roundtrip(agent)
                self.assertEqual(model_meta(restored), model_meta(agent))
                inputs = make_inputs()
                for a, b in zip(step(agent, *inputs)[:6], step(restored, *inputs)[:6]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_legacy_checkpoint_loads_as_parallel(self):
        # GAE 修复前的 checkpoint 没有结构 / 遮蔽字段：按并联、不遮蔽加载。
        kwargs = agent_kwargs_from_checkpoint({"num_actions": A})
        expected = dict(MODEL_META_DEFAULTS, num_actions=A)
        self.assertEqual(kwargs, expected)
        legacy = make_agent("parallel")
        ckpt = {"agent": legacy.state_dict(), "num_actions": A,
                "hidden_size": HIDDEN, "aux_heading": False,
                "vision_backbone": "small_cnn", "goal_type": "target_latlng"}
        restored = PPOAgent(**agent_kwargs_from_checkpoint(ckpt))
        restored.load_state_dict(ckpt["agent"])  # 严格加载
        self.assertEqual(restored.model_topology, "parallel")
        self.assertNotIn("status_fill", restored.state_dict())

    def test_wrong_topology_is_rejected(self):
        serial = make_agent("serial")
        with self.assertRaises(RuntimeError):
            make_agent("parallel").load_state_dict(serial.state_dict())

    def test_describe_agent(self):
        text = describe_agent(make_agent("serial", mask_position=True))
        self.assertIn("topology=serial", text)
        self.assertIn("masked_inputs=position", text)


class InputMaskTest(unittest.TestCase):

    def test_pose_masks_keep_goal(self):
        cases = {"mask_yaw": (0, 2), "mask_position": (2, 4)}
        for flag, (lo, hi) in cases.items():
            with self.subTest(flag=flag):
                agent = make_agent("parallel", **{flag: True}).eval()
                view, scalar, prev_r, prev_a = make_inputs()
                original = scalar.clone()
                _, state = agent.mask_inputs(view, scalar)
                self.assertTrue(torch.equal(scalar, original))   # 不改调用方张量
                self.assertTrue(torch.all(state[:, lo:hi] == 0))
                keep = [i for i in range(4) if not lo <= i < hi]
                self.assertTrue(torch.equal(state[:, keep], scalar[:, keep]))
                # 被遮蔽的切片怎么变都不影响输出；目标经纬度仍然影响输出。
                noisy = scalar.clone()
                noisy[:, lo:hi] = torch.rand(B, hi - lo)
                base = step(agent, view, scalar, prev_r, prev_a)[0]
                torch.testing.assert_close(
                    step(agent, view, noisy, prev_r, prev_a)[0], base)
                moved = scalar.clone()
                moved[:, 4:6] = 1.0 - moved[:, 4:6]
                self.assertGreater(
                    (step(agent, view, moved, prev_r, prev_a)[0] - base).abs().max().item(),
                    1e-6)

    def test_status_bar_mask(self):
        agent = make_agent("parallel", mask_status_rows=STATUS_ROWS).eval()
        view, scalar, prev_r, prev_a = make_inputs()
        original = view.clone()
        masked, _ = agent.mask_inputs(view, scalar)
        self.assertTrue(torch.equal(view, original))              # 不改调用方张量
        # 上方投影区逐像素不变，底部方向栏 = 黑色像素经 obs_to_batch 归一化后的值。
        self.assertTrue(torch.equal(masked[:, :, :-STATUS_ROWS], view[:, :, :-STATUS_ROWS]))
        black = {"view_image": np.zeros((3, H, W), dtype=np.uint8), "yaw": 0.0,
                 "latlng": np.zeros(2), "target_latlng": np.zeros(2)}
        black_t = obs_to_batch([black], torch.device("cpu"), (3, H, W))[0]
        expected = black_t[:, :, -STATUS_ROWS:, :].expand(B, -1, -1, -1)
        self.assertTrue(torch.equal(masked[:, :, -STATUS_ROWS:, :], expected))
        # 方向栏内容怎么变都不影响输出；上方画面变化会影响输出。
        bar_changed = view.clone()
        bar_changed[:, :, -STATUS_ROWS:, :] = torch.randn(B, 3, STATUS_ROWS, W)
        base = step(agent, view, scalar, prev_r, prev_a)[0]
        torch.testing.assert_close(step(agent, bar_changed, scalar, prev_r, prev_a)[0], base)
        top_changed = view.clone()
        top_changed[:, :, :STATUS_ROWS, :] += 1.0
        self.assertGreater(
            (step(agent, top_changed, scalar, prev_r, prev_a)[0] - base).abs().max().item(),
            1e-6)


if __name__ == "__main__":
    unittest.main()
