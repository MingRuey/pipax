"""
The pipeline trainer exposes interface to train models with pipeline model/pipeline parallelism,
which can inference/train on single batch data.
Internally, the pipeline schedule coordindates device hosts across multiple GPUs,
and has an small training loop on mini-batches for pipeline parallelism.
"""

from dataclasses import dataclass
from typing import Callable, Tuple, List
from queue import Queue

import jax
import jax.numpy as jnp
import jax.example_libraries.stax as stax
import jax.experimental.pjit

from src.model import mlp
from src.device_host import async_host


def accuracy(logit, label):
    predicted = jnp.argmax(logit, axis=-1)
    return jnp.mean(predicted == label[:, 0])


@dataclass
class NodesNums:
    nodes_per_layer: List[int]

    def __post_init__(self):
        self.nodes_per_layer = [int(n) for n in self.nodes_per_layer]


def PipelinedMlpTrainer(input_shape: Tuple[int],
                        device_nodes: List[NodesNums],
                        optimizer,
                        loss_fn: Callable,
                        activation=stax.Tanh,
                        pipelined: bool = True):
    """
    Train multilayer perceptron with pipeline parallel as proof of concept

    Args:
        input_shape (Tuple[int]): The input shape of the model
        nodes_per_layer (List[DeviceSpec]): The number of nodes on each GPU. One DeviceSpec for each GPU.
        optimizer: the optax.optimizer for updating weights
        loss_fn: the loss function with signature func(y_pred, y_true)
        activation (optional): The activation function of the dense layer.
        pipelined (optional): To enable pipeline parallelism or not.

    Returns:
        a tuple of callable (train, test),
        where train is the function for training on single batch (which has mini-batch loop inside).
        where test is the function for inferece.
    """
    gpus = jax.devices("gpu")

    assert len(gpus) >= 2, f"Not enough GPUs. Expect at least 2, get {len(gpus)}"
    assert len(device_nodes) == 2, f"{PipelinedMlpTrainer.__name__} currently only supports pipeline over 2 GPUs."

    w0, apply0 = mlp(input_shape, device_nodes[0].nodes_per_layer, activation=activation)
    w1, apply1 = mlp((device_nodes[0].nodes_per_layer[-1],), device_nodes[1].nodes_per_layer, activation=activation)

    w0, opt0, forward0, backward0, update0 = async_host(device=gpus[0], weights=w0, apply_fn=apply0, optimizer=optimizer)
    w1, opt1, forward1, backward1, update1 = async_host(device=gpus[1], weights=w1, apply_fn=apply1, optimizer=optimizer)

    loss_and_grads = jax.value_and_grad(lambda predict, label: loss_fn(predict, label))
    loss_and_grads = jax.jit(loss_and_grads, device=gpus[1])

    def init():
        q_predict = Queue()
        q_host0_fvjp = Queue()
        q_host1_fvjp = Queue()
        return (w0, opt0), (w1, opt1), (q_predict, q_host0_fvjp, q_host1_fvjp)

    def train_with_pipeline(batch_x, batch_y, pipe_state):
        m_size = batch_x.shape[0] // 2
        state0, state1, qs = pipe_state
        w0, opt0 = state0
        w1, opt1 = state1
        q_pred, q_fvjp0, q_fvjp1 = qs
        avg_loss = 0.0
        for f in range(2):
            x, y = batch_x[f*m_size:(f+1)*m_size, ...], batch_y[f*m_size:(f+1)*m_size, ...]
            out0, f_vjp0 = forward0(x, w0)
            predict, f_vjp1 = forward1(out0, w1)
            q_fvjp0.put(f_vjp0)
            q_fvjp1.put(f_vjp1)
            q_pred.put((predict, y))
        grad_collect0 = []
        grad_collect1 = []
        for b in range(2):
            loss, grads = loss_and_grads(*q_pred.get())
            back1, grads1 = backward1(grads, q_fvjp1.get())
            _, grads0 = backward0(back1, q_fvjp0.get())
            avg_loss += loss
            grad_collect1.append(grads1)
            grad_collect0.append(grads0)

        w0, opt0 = update0(grad_collect0, w0, opt0)
        w1, opt1 = update1(grad_collect1, w1, opt1)
        pipe_state = (w0, opt0), (w1, opt1), qs
        return avg_loss / 2, pipe_state

    def train_sequential(batch_x, batch_y, pipe_state):
        m_size = batch_x.shape[0] // 2
        state0, state1, qs = pipe_state
        w0, opt0 = state0
        w1, opt1 = state1
        q_pred, q_fvjp0, q_fvjp1 = qs
        avg_loss = 0.0
        grad_collect0 = []
        grad_collect1 = []
        for m in range(2):
            x, y = batch_x[m*m_size:(m+1)*m_size, ...], batch_y[m*m_size:(m+1)*m_size, ...]
            out0, f_vjp0 = forward0(x, w0)
            predict, f_vjp1 = forward1(out0, w1)
            q_fvjp0.put(f_vjp0)
            q_fvjp1.put(f_vjp1)
            q_pred.put((predict, y))
            loss, grads = loss_and_grads(*q_pred.get())
            back1, grads1 = backward1(grads, q_fvjp1.get())
            _, grads0 = backward0(back1, q_fvjp0.get())
            avg_loss += loss
            grad_collect1.append(grads1)
            grad_collect0.append(grads0)
        w0, opt0 = update0(grad_collect0, w0, opt0)
        w1, opt1 = update1(grad_collect1, w1, opt1)
        pipe_state = (w0, opt0), (w1, opt1), qs
        return avg_loss / 2, pipe_state

    train = train_with_pipeline if pipelined else train_sequential

    def test(x, y, pipe_state):
        state0, state1, qs = pipe_state
        w0, _ = state0
        w1, _ = state1
        logit, _ = forward1(forward0(x, w0)[0], w1)
        acc = accuracy(logit, y)
        test_loss = loss_fn(logit, y)
        return test_loss, acc

    return init, train, test
