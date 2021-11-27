from typing import Callable, Tuple, Any

import jax
from jax.core import ConcreteArray
from jaxlib.xla_client import Device
import optax

from src.model import mlp


def async_host(device: Device,
                weights: ConcreteArray,
                apply_fn: Callable,
                optimizer) -> Tuple[ConcreteArray, Any, Callable, Callable]:
    """A host on device whose weights NOT synchonizing with other devices

    This host is suitable for standard model parallelism.
    Each host connects with others in a sequential way.
    """

    weights = jax.device_put(weights, device=device)
    optimizer_state = optimizer.init(weights)

    def forward(forward_inputs: ConcreteArray,
                weights: ConcreteArray) -> Tuple[ConcreteArray, Callable]:
        outputs, f_vjp = jax.vjp(apply_fn, weights, forward_inputs)
        return outputs, f_vjp
    forward = jax.jit(forward, device=device)

    def backward(backward_inputs: ConcreteArray,
                 weights: ConcreteArray,
                 optimizer_state,
                 f_vjp: Callable) -> Tuple[ConcreteArray, ConcreteArray, Any]:
        # The magic for calculating the gradients of loss w.r.t weights on the device:
        # chain the gradients from previous layer to the current
        grads_wrt_weights, grads_wrt_foward_inputs = f_vjp(backward_inputs)

        updates, optimizer_state = optimizer.update(grads_wrt_weights, optimizer_state)
        weights = optax.apply_updates(updates, weights)
        return grads_wrt_foward_inputs, weights, optimizer_state
    backward = jax.jit(backward, device=device)

    return weights, optimizer_state, forward, backward
