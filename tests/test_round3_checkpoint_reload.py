import torch

def test_tracker_checkpoint_container_roundtrip(tmp_path):
    source = torch.nn.Linear(4, 3)
    path = tmp_path / "checkpoint.pt"
    torch.save({"tracker_state": source.state_dict(), "optimizer": None, "scheduler": None, "global_step": 7}, path)
    target = torch.nn.Linear(4, 3)
    payload = torch.load(path, weights_only=False)
    target.load_state_dict(payload["tracker_state"], strict=True)
    x = torch.randn(2, 4)
    assert torch.equal(source(x), target(x))
    assert payload["global_step"] == 7
