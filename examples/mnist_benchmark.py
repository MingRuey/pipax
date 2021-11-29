"""Benchmark model/pipline parallelism on MNIST data"""
class TestBenchmarkCrossDevice:

    def test_simple_model_parallelism(self):
        """Split the MLP across devices, and use naive scheduling, i.e. without Pipeline parallelism"""
        pass

    def test_gpipe_scheduling(self):
        """Split the MLP across devices, and use GPipe scheduling

        Reference:
            GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
            https://arxiv.org/abs/1811.06965
        """
        pass


if __name__ == "__main__":
    pass
