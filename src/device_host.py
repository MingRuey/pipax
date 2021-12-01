from typing import Callable, Tuple, Any

import jax
from jax._src.tree_util import tree_reduce, tree_map
from jax.core import ConcreteArray
from jax.tree_util import tree_multimap
from jaxlib.xla_client import Device
import optax


def async_host(device: Device,
               weights: ConcreteArray,
               apply_fn: Callable,
               optimizer) -> Tuple[ConcreteArray, Any, Callable, Callable]:
    """A host on device whose weights NOT synchronizing with other devices

    This host is suitable for standard model parallelism:
        the model is divided into sequential stages, and each stage running
        its forward/backward without synchronization with other stages.
    """

    weights = jax.device_put(weights, device=device)
    optimizer_state = optimizer.init(weights)
    optimizer_state= jax.device_put(optimizer_state, device=device)

    def forward(forward_inputs: ConcreteArray,
                weights: ConcreteArray) -> Tuple[ConcreteArray, Callable]:
        outputs, f_vjp = jax.vjp(apply_fn, weights, forward_inputs)
        return outputs, f_vjp
    forward = jax.jit(forward, device=device)

    def backward(backward_inputs: ConcreteArray,
                 f_vjp: Callable) -> Tuple[ConcreteArray, ConcreteArray, Any]:
        # The magic for calculating the gradients of loss w.r.t weights on the device:
        # chain the gradients from previous layer to the current
        grads_wrt_weights, grads_wrt_foward_inputs = f_vjp(backward_inputs)
        return grads_wrt_foward_inputs, grads_wrt_weights
    backward = jax.jit(backward, device=device)

    def update(grads_collection, weights, optimizer_state):
        grads = tree_multimap(lambda x, y: x + y, *grads_collection)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        weights = optax.apply_updates(updates, weights)
        return weights, optimizer_state
    update = jax.jit(update, device=device)

    return weights, optimizer_state, forward, backward, update
