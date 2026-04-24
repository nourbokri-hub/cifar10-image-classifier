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