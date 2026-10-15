"""Matched shared-adapter contracts; Torch execution belongs on verified compute."""

import copy
import importlib.util
import io
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from semflow.models.adapters import (
        LoRALinear,
        enable_shared_adapter_training,
        inject_shared_lora,
        named_shared_adapter_parameters,
    )

    class TinyPolicy(nn.Module):
        """Only the registered native q/v topology is represented here."""

        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.language_model = nn.Module()
            self.backbone.language_model.layers = nn.ModuleList()
            for _ in range(2):
                layer = nn.Module()
                layer.self_attn = nn.Module()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(layer.self_attn, name, nn.Linear(16, 16))
                self.backbone.language_model.layers.append(layer)
            self.head = nn.Linear(16, 7)

        def forward(self, inputs):
            for layer in self.backbone.language_model.layers:
                attention = layer.self_attn
                inputs = attention.o_proj(torch.tanh(attention.q_proj(inputs)) + attention.v_proj(inputs))
            return self.head(inputs)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional cloud training dependency")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_zero_residual_preserves_output_and_dtype(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            base = nn.Linear(16, 12, bias=True, dtype=dtype).eval()
            inputs = torch.randn(2, 3, 16, dtype=dtype)
            expected = base(inputs).detach()
            adapted = LoRALinear(base)
            torch.testing.assert_close(adapted(inputs), expected, rtol=0, atol=0)
            self.assertEqual(adapted.lora_A.dtype, dtype)
            self.assertEqual(adapted.lora_B.device, base.weight.device)
            self.assertFalse(adapted.training)
            self.assertTrue(all(not value.requires_grad for value in adapted.base.parameters()))

    def test_rank_and_base_validation(self):
        for rank in (0, -1, True, 1.5, 17):
            with self.assertRaises(ValueError):
                LoRALinear(nn.Linear(16, 16), rank=rank)
        with self.assertRaises(TypeError):
            LoRALinear(nn.Identity())
        for rank in (0, 4, 8.0, True):
            with self.assertRaisesRegex(ValueError, "rank 8"):
                inject_shared_lora(TinyPolicy(), rank=rank)

    def test_injection_preserves_native_function_and_exact_projection_set(self):
        policy = TinyPolicy()
        inputs = torch.randn(2, 16)
        expected = policy(inputs).detach()
        paths = inject_shared_lora(policy)
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(path.endswith(("q_proj", "v_proj")) for path in paths))
        torch.testing.assert_close(policy(inputs), expected, rtol=0, atol=0)
        for layer in policy.backbone.language_model.layers:
            self.assertIsInstance(layer.self_attn.k_proj, nn.Linear)
            self.assertIsInstance(layer.self_attn.o_proj, nn.Linear)
        names = [name for name, _ in named_shared_adapter_parameters(policy)]
        self.assertEqual(len(names), 8)
        self.assertTrue(all(name.endswith((".lora_A", ".lora_B")) for name in names))

    def test_frozen_base_and_readout_preserve_input_gradients(self):
        base = nn.Linear(16, 16)
        reference = copy.deepcopy(base)
        adapted = LoRALinear(base)
        adapted.requires_grad_(False)
        inputs = torch.randn(3, 16, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        adapted(inputs).square().mean().backward()
        reference(reference_inputs).square().mean().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        self.assertGreater(inputs.grad.norm().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in adapted.parameters()))

    def test_optimizer_updates_only_shared_adapters_through_frozen_head(self):
        policy = TinyPolicy()
        inject_shared_lora(policy)
        trainable = enable_shared_adapter_training(policy)
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=0.01)
        before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            policy(torch.randn(3, 16)).square().mean().backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for _, parameter in trainable))
            optimizer.step()
        changed = {name for name, parameter in policy.named_parameters() if not torch.equal(parameter, before[name])}
        selected = {name for name, _ in trainable}
        self.assertTrue(changed)
        self.assertEqual(changed, selected)
        self.assertTrue(all(parameter.grad is None for name, parameter in policy.named_parameters() if name not in selected))

    def test_adaptation_freezes_entire_wrapper_and_clears_old_gradients(self):
        policy = TinyPolicy()
        inject_shared_lora(policy)
        wrapper = nn.Module()
        wrapper.native_policy = policy
        wrapper.readout = nn.Linear(7, 7)
        for parameter in wrapper.parameters():
            parameter.grad = torch.ones_like(parameter)
        selected = enable_shared_adapter_training(wrapper, policy=policy)
        self.assertTrue(all(name.startswith("native_policy.backbone.") for name, _ in selected))
        self.assertEqual({name for name, parameter in wrapper.named_parameters() if parameter.requires_grad},
                         {name for name, _ in selected})
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.parameters()))
        self.assertTrue(all(not child.training for child in wrapper.modules()))
        wrapper.readout(policy(torch.randn(2, 16))).square().mean().backward()
        self.assertGreater(sum(parameter.grad.norm().item() for _, parameter in selected), 0)
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.readout.parameters()))

    def test_matched_arms_restore_identical_common_state_and_trainable_set(self):
        source = TinyPolicy()
        inject_shared_lora(source)
        # Simulate source-calibrated nonzero adapter weights before cloning.
        with torch.no_grad():
            for _, parameter in named_shared_adapter_parameters(source):
                parameter.add_(torch.randn_like(parameter) * 0.01)
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        arms = []
        for _ in range(2):
            arm = TinyPolicy()
            inject_shared_lora(arm)
            arm.load_state_dict(checkpoint, strict=True)
            names = tuple(name for name, _ in enable_shared_adapter_training(arm))
            arms.append((arm, names))
        first, second = arms[0][0], arms[1][0]
        self.assertEqual(arms[0][1], arms[1][1])
        inputs = torch.randn(2, 16)
        torch.testing.assert_close(first(inputs), source(inputs), atol=0, rtol=0)
        torch.testing.assert_close(first(inputs), second(inputs), atol=0, rtol=0)
        for name, parameter in first.named_parameters():
            torch.testing.assert_close(parameter, dict(second.named_parameters())[name], atol=0, rtol=0)
            self.assertIsNot(parameter, dict(second.named_parameters())[name])

    def test_invalid_late_projection_does_not_partially_inject(self):
        policy = TinyPolicy()
        policy.backbone.language_model.layers[-1].self_attn.v_proj = nn.Identity()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            inject_shared_lora(policy)
        first = policy.backbone.language_model.layers[0].self_attn.q_proj
        self.assertIsInstance(first, nn.Linear)
        self.assertTrue(first.weight.requires_grad)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            inject_shared_lora(nn.Module())

    def test_duplicate_wrapping_and_parameter_aliases_are_rejected(self):
        policy = TinyPolicy()
        inject_shared_lora(policy)
        with self.assertRaisesRegex(ValueError, "already wrapped"):
            inject_shared_lora(policy)
        alias = TinyPolicy()
        layer = alias.backbone.language_model.layers[0]
        layer.self_attn.v_proj.weight = layer.self_attn.q_proj.weight
        with self.assertRaisesRegex(ValueError, "alias"):
            inject_shared_lora(alias)
        policy.adapter_alias = policy.backbone.language_model.layers[0].self_attn.q_proj.lora_A
        with self.assertRaisesRegex(ValueError, "exactly one"):
            named_shared_adapter_parameters(policy)

    def test_adaptation_rejects_unregistered_external_policy(self):
        policy = TinyPolicy()
        inject_shared_lora(policy)
        other = nn.Linear(16, 16)
        with self.assertRaisesRegex(ValueError, "registered inside"):
            enable_shared_adapter_training(other, policy=policy)
        self.assertTrue(other.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
