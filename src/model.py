from typing import Tuple, List

import jax
import jax.example_libraries.stax as stax
from jax.nn.initializers import xavier_normal, normal

from src.random_seed import STATIC_KEY


def mlp(input_shape: Tuple[int], nodes_per_layer: List[int], activation=stax.Tanh):
    """Simple multilayer perceptron for demonstration purpose"""
    layers = []
    for n in nodes_per_layer[:-1]:
        layers.append(stax.Dense(n, W_init=xavier_normal(), b_init=normal()))
        layers.append(activation)
    layers.append(stax.Dense(nodes_per_layer[-1], W_init=xavier_normal(), b_init=normal()))
    init, apply = stax.serial(*layers)
    out_shape, weights = init(STATIC_KEY, input_shape)
    return weights, apply
