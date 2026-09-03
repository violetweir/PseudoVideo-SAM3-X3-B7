import importlib.util
from pathlib import Path

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
spec = importlib.util.spec_from_file_location("round3_train", ROOT / "scripts/train_round3_tracker.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_memory_policy_is_exact():
    assert module.is_memory_parameter("maskmem_backbone.out_proj.weight")
    assert module.is_memory_parameter("transformer.encoder.layers.0.linear1.weight")
    assert module.is_memory_parameter("maskmem_tpos_enc")
    assert module.is_memory_parameter("no_mem_embed")
    assert not module.is_memory_parameter("sam_mask_decoder.iou_prediction_head.layers.0.weight")
    assert not module.is_memory_parameter("sam_prompt_encoder.no_mask_embed.weight")
    assert not module.is_memory_parameter("obj_ptr_proj.layers.0.weight")
