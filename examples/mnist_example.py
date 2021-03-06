"""
Baseline benchmark with MNIST and simple MLP

It runs on single device, without any model/pipeline parallelism
We select a computing-bound setup (a MLP with memory size around 40MB)
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.example_libraries.stax import Relu
import numpy as np
import optax
import pytest

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import mlp
from examples.load_data import MNIST

def loss_fn(logit, label):
    label = jax.nn.one_hot(label, num_classes=10)[:, 0, :]
    logloss = -jnp.sum(jax.nn.log_softmax(logit) * label)
    return jnp.mean(logloss) / logit.shape[0]

def accuracy(logit, label):
    predicted = jnp.argmax(logit, axis=-1)
    return jnp.mean(predicted == label[:, 0])

def random_shuffle(images, labels):
    assert images.shape[0] == labels.shape[0]
    n_data = images.shape[0]
    indices = np.random.permutation(n_data)
    return images[indices, ...], labels[indices, ...]


class TestBenchmarkOnSingleDevice:
    """A simple example for training MLP on MNIST (without model/pipeline parallelism)"""

    def test_simple_mlp_with_adam(self):
        epoch = 50
        batch = 4096
        weights, apply = mlp(input_shape=(28 * 28,), nodes_per_layer=[1024] * 16 + [10], activation=Relu)
        optimizer = optax.adam(learning_rate=0.001)
        opt_state = optimizer.init(weights)

        @jax.jit
        def single_iteration(imgs, labels, weights, opt_state):
            loss, grads = jax.value_and_grad(lambda w, x, y: loss_fn(apply(w, x), y))(weights, imgs, labels)
            update, opt_state = optimizer.update(grads, opt_state)
            weights = optax.apply_updates(weights, update)
            return weights, opt_state, loss

        imgs, labels = MNIST.get_all_train()
        imgs_test, labels_test = MNIST.get_all_test()
        imgs_test, labels_test = imgs_test.reshape(imgs_test.shape[0], -1), labels_test.reshape(labels_test.shape[0], -1)

        batch_per_epoch = imgs.shape[0] // batch
        print("\n")
        for e in range(epoch):
            imgs, labels = random_shuffle(imgs, labels)
            imgs = jax.device_put(imgs, device=jax.devices("gpu")[0])
            labels = jax.device_put(labels, device=jax.devices("gpu")[0])
            avg_loss = 0.0
            start = time.time()
            for b in range(batch_per_epoch):
                x, y = imgs[b * batch: (b + 1) * batch, ...], labels[b * batch: (b + 1) * batch, ...]
                weights, opt_state, loss = single_iteration(x, y, weights, opt_state)
                avg_loss += loss
            elapsed = time.time() - start

            logit = apply(weights, imgs_test)
            acc = accuracy(logit, labels_test)
            avg_loss /= batch_per_epoch
            test_loss = loss_fn(logit, labels_test)
            print(f"Epoch {e} ({batch_per_epoch * batch} images/{elapsed:.4f} sec): Train loss {avg_loss:.4f}, Test loss {test_loss:.4f}, Acc On Test: {acc:.3f}")


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__, "--durations=0"])
