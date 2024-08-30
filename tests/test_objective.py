import os
import pytest
import pandas as pd
from copy import deepcopy
import torch

from feat_viz.objective import *
from feat_viz.utils import load_model
import numpy as np

random_image = torch.randn(1, 3, 244, 244)


@pytest.fixture
def model():
    return load_model("inception_v3").eval()


@pytest.fixture
def mixed7c_layer() -> np.ndarray:
    model = load_model("inception_v3").eval()
    hook = Hook("Mixed_7c")
    hook(model)
    model(random_image)
    f = hook.feature.detach().cpu().numpy()
    return deepcopy(f)


def create_hook_and_return_feature(model, **kwargs) -> np.ndarray:
    hook = Hook(**kwargs)
    hook(model)
    model(random_image)
    f = hook.feature.detach().cpu().numpy()
    return f


def test_whole_layer(mixed7c_layer, model):
    f = create_hook_and_return_feature(model, layer_name="Mixed_7c")
    assert f.shape == (2048, 6, 6), f"Expected shape: {(2048, 6, 6)}, got {f.shape}"
    assert np.allclose(f, mixed7c_layer), "Features are not equal"


def test_whole_channel_only_single(mixed7c_layer, model):
    f = create_hook_and_return_feature(model, layer_name="Mixed_7c", channel_number=1)
    assert f.shape == (1, 6, 6), f"Expected shape: {(1, 6, 6)}, got {f.shape}"
    assert np.allclose(f, mixed7c_layer[1, ...]), "Features are not equal"


def test_whole_channel_only_list(mixed7c_layer, model):
    f = create_hook_and_return_feature(
        model, layer_name="Mixed_7c", channel_number=[1, 2, 3]
    )
    assert f.shape == (3, 6, 6), f"Expected shape: {(3, 6, 6)}, got {f.shape}"
    assert np.allclose(f, mixed7c_layer[1:4, ...]), "Features are not equal"


def test_whole_channel_only_tuple(mixed7c_layer, model):
    f = create_hook_and_return_feature(
        model, layer_name="Mixed_7c", channel_number=(10, 20)
    )
    assert f.shape == (10, 6, 6), f"Expected shape: {(10, 6, 6)}, got {f.shape}"
    assert np.allclose(f, mixed7c_layer[10:20, ...]), "Features are not equal"


def test_whole_channel_single_neuron(mixed7c_layer, model):
    f = create_hook_and_return_feature(
        model,
        layer_name="Mixed_7c",
        channel_number=(10, 20),
        neuron_number=1,
    )
    assert f.shape == (10, 1, 1), f"Expected shape: {(10, 1, 1)}, got {f.shape}"
    assert np.allclose(
        f, mixed7c_layer[10:20, 0:1, 1:2]
    ), "Features are not equal"  # row 0, column 1


def test_whole_layer_negative_index(mixed7c_layer, model):
    # Testing with a negative index to access the last channel
    f = create_hook_and_return_feature(model, layer_name="Mixed_7c", channel_number=-1)
    assert f.shape == (1, 6, 6), "Expected shape: (1, 6, 6), got {}".format(f.shape)
    assert np.allclose(f, mixed7c_layer[-1, ...]), "Features are not equal"


def test_multiple_neurons_single_channel(mixed7c_layer, model):
    # Testing extraction of multiple specific neurons from a single channel
    f = create_hook_and_return_feature(
        model,
        layer_name="Mixed_7c",
        channel_number=5,
        neuron_number=(
            (0, 1),
            (2, 3),
        ),  # Extracting two neurons: (row 0, col 1) and (widht 2, and 3)
    )
    expected_shape = (
        1,
        2,
        3,
    )
    assert f.shape == expected_shape, f"Expected shape: {expected_shape}, got {f.shape}"
    assert np.allclose(f, mixed7c_layer[5, 0:2, 1:4]), "Features are not equal"


def test_multiple_neurons_all_channel(mixed7c_layer, model):
    # Testing extraction of multiple specific neurons from a single channel
    f = create_hook_and_return_feature(
        model,
        layer_name="Mixed_7c",
        neuron_number=(
            (0, 1),
            (2, 3),
        ),  # Extracting two neurons: (row 0, col 1) and (widht 2, and 3)
    )
    expected_shape = (
        2048,  # number of channels in mixed7c_layer
        2,
        3,
    )
    assert f.shape == expected_shape, f"Expected shape: {expected_shape}, got {f.shape}"
    assert np.allclose(f, mixed7c_layer[:, 0:2, 1:4]), "Features are not equal"


# def test_entire_layer_as_vector(mixed7c_layer, model):
#     # Testing flattening the entire layer into a single vector
#     f = create_hook_and_return_feature(model, layer_name="Mixed_7c", flatten=True)
#     expected_shape = (2048 * 6 * 6,)  # Flattened vector shape
#     assert f.shape == expected_shape, "Expected shape: {}, got {}".format(
#         expected_shape, f.shape
#     )
#     assert np.allclose(f, mixed7c_layer.flatten()), "Features are not equal"


# def test_specific_neurons_across_channels(mixed7c_layer, model):
#     # Testing extraction of a specific neuron across multiple channels
#     f = create_hook_and_return_feature(
#         model,
#         layer_name="Mixed_7c",
#         channel_number=[0, 100, 500],  # Selecting three channels
#         neuron_number=(2, 4),  # Selecting a specific neuron (row 2, col 4)
#     )
#     expected_shape = (3, 1, 1)  # Three channels, one neuron
#     assert f.shape == expected_shape, "Expected shape: {}, got {}".format(
#         expected_shape, f.shape
#     )
#     assert np.allclose(
#         f, mixed7c_layer[[0, 100, 500], 2:3, 4:5]
#     ), "Features are not equal"


# def test_channel_range(mixed7c_layer, model):
#     # Testing extraction of a range of channels
#     f = create_hook_and_return_feature(
#         model,
#         layer_name="Mixed_7c",
#         channel_number=range(100, 200),  # Selecting a range of channels
#     )
#     expected_shape = (100, 6, 6)  # 100 channels, each with a 6x6 grid
#     assert f.shape == expected_shape, "Expected shape: {}, got {}".format(
#         expected_shape, f.shape
#     )
#     assert np.allclose(f, mixed7c_layer[100:200, ...]), "Features are not equal"
