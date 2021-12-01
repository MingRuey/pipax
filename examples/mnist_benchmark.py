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
from src.pipelined_trainer import PipelinedMlpTrainer, NodesNums
from examples.load_data import MNIST
from examples.mnist_example import loss_fn, accuracy, random_shuffle


class TestBenchmarkCrossDevice:

    @staticmethod
    def train(pipelined: bool, epoch: int=50, batch: int=4096):
        optimizer = optax.adam(learning_rate=0.001)

        init, train, test = PipelinedMlpTrainer(
            input_shape=(28*28,), device_nodes=[NodesNums([1024] * 8), NodesNums([1024] * 8 + [10])],
            optimizer=optimizer, loss_fn=loss_fn, activation=Relu, pipelined=pipelined)
        pipe_state = init()

        imgs, labels = MNIST.get_all_train()
        imgs_test, labels_test = MNIST.get_all_test()
        imgs_test, labels_test = imgs_test.reshape(imgs_test.shape[0], -1), labels_test.reshape(labels_test.shape[0], -1)

        print("\n")
        batch_per_epoch = imgs.shape[0] // batch
        for e in range(epoch):
            imgs, labels = random_shuffle(imgs, labels)
            imgs = jax.device_put(imgs, device=jax.devices("gpu")[0])
            labels = jax.device_put(labels, device=jax.devices("gpu")[1])
            avg_loss = 0.0
            start = time.time()
            for b in range(batch_per_epoch):
                x, y = imgs[b * batch: (b + 1) * batch, ...], labels[b * batch: (b + 1) * batch, ...]
                loss, pipe_state = train(x, y, pipe_state)
                avg_loss += loss
            elapsed = time.time() - start

            test_loss, acc = test(imgs_test, labels_test, pipe_state)
            avg_loss /= batch_per_epoch
            print(f"Epoch {e} ({batch_per_epoch * batch} images/{elapsed:.4f} sec): Train loss {avg_loss:.4f}, Test loss {test_loss:.4f}, Acc On Test: {acc:.3f}")


    def test_simple_model_parallelism(self):
        """Split the MLP across devices, and use naive scheduling, i.e. without Pipeline parallelism"""
        self.train(False)

    def test_gpipe_scheduling(self):
        """Split the MLP across devices, and use GPipe-like scheduling

        Reference:
            GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
            https://arxiv.org/abs/1811.06965
        """
        self.train(True)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__, "--durations=0"])
