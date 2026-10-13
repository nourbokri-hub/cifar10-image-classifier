import pytest
import torch

from src.model import get_model


def test_model_creation():
    model = get_model(num_classes=10)
    assert model is not None


def test_model_output_shape():
    model = get_model(num_classes=10)
    model.eval()

    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 10)


def test_training_refuses_to_overwrite_existing_checkpoint(tmp_path, monkeypatch):
    import src.training as training

    checkpoint = tmp_path / "model.pt"
    original_contents = b"existing checkpoint"
    checkpoint.write_bytes(original_contents)
    monkeypatch.setattr(training, "MODEL_PATH", checkpoint)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        training.main()

    assert checkpoint.read_bytes() == original_contents
