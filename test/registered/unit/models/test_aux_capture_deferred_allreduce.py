"""Aux hidden states captured between decoder layers for a draft model are a
layer's output plus the residual. When the layer left its FFN all-reduce to the
next layer, that output is one rank's partial sum until the reduction runs."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.communicator import LayerCommunicator, UnreducedOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TP_SIZE = 2
HIDDEN = 4
TOKENS = 3
NUM_LAYERS = 4
CAPTURED = [1, 2, 3]


def all_reduce(hidden_states):
    # Every rank holds the same partial sum in this single-process stand-in.
    return hidden_states * TP_SIZE


class DeferringLayer(nn.Module):
    """One TP rank's view of a decoder layer whose FFN output sums to one. It
    completes a reduction left by the previous layer, and leaves its own to the
    next layer unless it is the last."""

    def __init__(self, is_last_layer):
        super().__init__()
        self.is_last_layer = is_last_layer
        self.layer_communicator = LayerCommunicator.__new__(LayerCommunicator)

    def forward(
        self, positions=None, hidden_states=None, forward_batch=None, residual=None, **_
    ):
        if isinstance(hidden_states, UnreducedOutput):
            hidden_states = all_reduce(hidden_states.partial)
        residual = hidden_states if residual is None else hidden_states + residual
        if self.is_last_layer:
            return torch.ones_like(residual), residual
        return UnreducedOutput(torch.full_like(residual, 1 / TP_SIZE)), residual


class SumNorm(nn.Module):
    def forward(self, hidden_states, residual=None, post_residual_addition=None):
        if residual is None:
            return hidden_states
        return hidden_states + residual, residual


def stub_model(module, cls):
    model_cls = getattr(importlib.import_module(module), cls)
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    attrs = dict(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        layers=[DeferringLayer(i == NUM_LAYERS - 1) for i in range(NUM_LAYERS)],
        layers_to_capture=CAPTURED,
        start_layer=0,
        end_layer=NUM_LAYERS,
        hidden_size=HIDDEN,
        norm=SumNorm(),
        use_hf_deepstack_order=False,
    )
    for name, value in attrs.items():
        object.__setattr__(model, name, value)
    return model


MODELS = (
    ("sglang.srt.models.qwen3_vl_moe", "Qwen3MoeLLMModel"),
    ("sglang.srt.models.gpt_oss", "GptOssModel"),
    ("sglang.srt.models.laguna", "LagunaModel"),
    ("sglang.srt.models.bailing_moe", "BailingMoEModel"),
)


class TestAuxCaptureOnDeferredReduction(CustomTestCase):
    def setUp(self):
        patcher = patch(
            "sglang.srt.layers.communicator.deferred_post_experts_all_reduce",
            all_reduce,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_captures_see_the_completed_sum(self):
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False)
        )
        for module, cls in MODELS:
            with self.subTest(model=cls):
                hidden_states, aux = stub_model(module, cls).forward(
                    input_ids=None,
                    positions=None,
                    forward_batch=forward_batch,
                    input_embeds=torch.zeros(TOKENS, HIDDEN),
                )
                # Captured before layer i: layer i - 1's summed output (1) plus
                # the residual accumulated over layers 0 .. i - 2.
                self.assertEqual(len(aux), len(CAPTURED))
                for i, captured in zip(CAPTURED, aux):
                    torch.testing.assert_close(
                        captured, torch.full((TOKENS, HIDDEN), float(i))
                    )
                torch.testing.assert_close(
                    hidden_states, torch.full((TOKENS, HIDDEN), float(NUM_LAYERS))
                )


if __name__ == "__main__":
    unittest.main()
