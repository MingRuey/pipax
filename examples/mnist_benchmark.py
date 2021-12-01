"""
Benchmark model/pipline parallelism on MNIST data

We select a computing-bound setup (a MLP with memory size around 40MB)
"""
import sys
import time
from pathlib import Path
from itertools import cycle
from queue import Queue
from typing import Iterable

import jax
from jax.example_libraries.stax import Relu
import optax
import pytest

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import mlp
from src.device_host import async_host
from examples.load_data import MNIST
from examples.mnist_example import loss_fn, accuracy, random_shuffle


class TestBenchmarkCrossDevice:

    @staticmethod
    def train(scheduling: Iterable[str], epoch: int=50, batch: int=4096):
        gpus = jax.devices("gpu")
        assert len(gpus) >= 2, "Not enough GPU for testing parallelism. Required 2."
        q_back_grads = Queue()
        q_host0_fvjp = Queue()
        q_host1_fvjp = Queue()

        optimizer = optax.adam(learning_rate=0.001)
        calculcate_loss = jax.jit(jax.value_and_grad(lambda predict, label: loss_fn(predict, label)), device=gpus[1])

        w0, apply0 = mlp(input_shape=(28 * 28,), nodes_per_layer=[1024] * 8, activation=Relu)
        w0, opt0, forward0, backward0 = async_host(device=gpus[0], weights=w0, apply_fn=apply0, optimizer=optimizer)

        w1, apply1 = mlp(input_shape=(1024,), nodes_per_layer=[1024] * 8 + [10], activation=Relu)
        w1, opt1, forward1, backward1 = async_host(device=gpus[1], weights=w1, apply_fn=apply1, optimizer=optimizer)

        imgs, labels = MNIST.get_all_train()
        imgs_test, labels_test = MNIST.get_all_test()
        imgs_test, labels_test = imgs_test.reshape(imgs_test.shape[0], -1), labels_test.reshape(labels_test.shape[0], -1)

        print("\n")
        batch_per_epoch = imgs.shape[0] // batch
        mini_batch = batch // 2
        for e in range(epoch):
            imgs, labels = random_shuffle(imgs, labels)
            elapsed = 0.0
            for b in range(batch_per_epoch):
                bx, by = imgs[b * batch: (b + 1) * batch, ...], labels[b * batch: (b + 1) * batch, ...]
                # Mini-batch loop
                m = 0
                for job in scheduling:
                    if job == "forward":
                        x, y = bx[m * mini_batch: (m + 1) * mini_batch, ...], by[m * mini_batch: (m + 1) * mini_batch, ...]
                        m += 1
                        x = jax.device_put(x, device=jax.devices("gpu")[0])
                        y = jax.device_put(y, device=jax.devices("gpu")[0])

                        start = time.time()
                        out0, f_vjp0 = forward0(x, w0)
                        predict, f_vjp1 = forward1(out0, w1)
                        loss, grads = calculcate_loss(predict, y)

                        q_host0_fvjp.put(f_vjp0)
                        q_host1_fvjp.put(f_vjp1)
                        q_back_grads.put(grads)
                        elapsed += time.time() - start
                    else:
                        start = time.time()
                        back1, w1, opt1 = backward1(q_back_grads.get(), w1, opt1, q_host1_fvjp.get())
                        _, w0, opt0 = backward0(back1, w0, opt0, q_host0_fvjp.get())
                        elapsed += time.time() - start

            logit, _ = forward1(forward0(imgs_test, w0)[0], w1)
            acc = accuracy(logit, labels_test)
            test_loss = loss_fn(logit, labels_test)
            print(f"Epoch {e} ({batch_per_epoch * batch} images/{elapsed:.4f} sec): Test loss {test_loss}, Acc On Test: {acc}")


    def test_simple_model_parallelism(self):
        """Split the MLP across devices, and use naive scheduling, i.e. without Pipeline parallelism"""
        scheduling = ["forward", "backward", "forward", "backward"]
        self.train(scheduling)

    def test_gpipe_scheduling(self):
        """Split the MLP across devices, and use GPipe-like scheduling

        Reference:
            GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
            https://arxiv.org/abs/1811.06965
        """
        scheduling = ["forward", "forward", "backward", "backward"]
        self.train(scheduling)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__, "--durations=0"])
