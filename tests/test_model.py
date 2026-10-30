"""Tests for the ResNet18 model factory (offline)."""

import socket
from urllib.error import URLError

import pytest
import torch
import torchvision.models._api as weights_api
from torch import nn

from medvision.data.constants import NUM_CLASSES
from medvision.models.resnet import build_resnet18


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not access the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_forward_pass_returns_logits_for_seven_classes() -> None:
    model = build_resnet18(pretrained=False).eval()

    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))

    assert logits.shape == (2, NUM_CLASSES)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == NUM_CLASSES
    # Raw logits: no softmax layer, so rows are not probability distributions.
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2))


def test_random_initialisation_never_loads_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Weights must not be loaded when pretrained=False.")

    monkeypatch.setattr(weights_api, "load_state_dict_from_url", fail)

    build_resnet18(pretrained=False)


def test_pretrained_failure_is_raised_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(weights_api, "load_state_dict_from_url", unavailable)

    with pytest.raises(RuntimeError, match="Could not load pretrained ResNet18 weights"):
        build_resnet18(pretrained=True)


def test_invalid_class_count_fails() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        build_resnet18(pretrained=False, num_classes=1)
