import importlib.util
from pathlib import Path

import torch

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
spec = importlib.util.spec_from_file_location("round3_lora", ROOT / "scripts/round3_lora.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_lora_linear_is_identity_at_initialization_and_freezes_base():
    base = torch.nn.Linear(8, 5)
    inputs = torch.randn(4, 8)
    expected = base(inputs).detach()
    wrapped = module.LoRALinear(base, rank=4, alpha=8, dropout=0.0)
    assert torch.equal(wrapped(inputs), expected)
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_A.requires_grad and wrapped.lora_B.requires_grad


def test_target_patterns_are_exact():
    assert module.target_kind("transformer.encoder.layers.0.self_attn.q_proj") == "memory_attention"
    assert module.target_kind("transformer.encoder.layers.3.cross_attn_image.out_proj") == "memory_attention"
    assert module.target_kind("sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj") == "mask_decoder_cross_attention"
    assert module.target_kind("sam_mask_decoder.transformer.final_attn_token_to_image.k_proj") == "mask_decoder_cross_attention"
    assert module.target_kind("transformer.encoder.layers.0.linear1") is None
    assert module.target_kind("sam_mask_decoder.transformer.layers.0.self_attn.q_proj") is None


def test_update_norm_reads_parameter_pairs():
    tracker = torch.nn.Module()
    tracker.proj = module.LoRALinear(torch.nn.Linear(8, 5), rank=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        tracker.proj.lora_B.fill_(0.01)
    norms = module.lora_update_norms(tracker)
    assert norms["total"] > 0
    assert norms["maximum"] > 0
