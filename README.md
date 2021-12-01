# Pipax
Experiments on pipeline parallelism using Jax.
It's still in the drafting/proof of concept stage.

## Introduction
Pipeline parallelism is an optimized model parallelism, where stages on difference

## Dependency
The environment for testing the project:
```
python >= 3.8.10
jax==0.2.25
jaxlib==0.1.74+cuda11.cudnn805
optax==0.1.0
pytest==6.2.4
tensorboard==2.6.0
```
You may want to change the CUDA/cudnn version to match your device.

## Test Runs
Baseline - multilayer perceptron on MNIST, single GPU
```bash
python examples/mnist_example.py
```

Experiments - multilayer perceptron on MNIST, with
1. Simple model parallelism without pipeline
2. Model parallelism with pipeline parallelism

```bash
python examples/mnist_benchmark.py
```

## Results
Initial attempts show rouhgly 20% speed-up using pipeline (on a machine with multiple RTX5000).

## ToDo:
- Detailed performance comparison. Even better with GPU profiling.
- Try reducing the Python for-loop overhead in pipeline.
