from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT.joinpath("examples").joinpath("data")

def extract_image(file: str):
    """Extract the (normalized) image from the idx file"""
    file = DATA_DIR.joinpath(file)
    with open(file, "rb") as stream:
        stream.read(4) # header
        n_images = int.from_bytes(stream.read(4), byteorder="big")
        n_row = int.from_bytes(stream.read(4), byteorder="big")
        n_col = int.from_bytes(stream.read(4), byteorder="big")
        buf = stream.read(n_row * n_col * n_images)
        data = np.frombuffer(buf, np.uint8).astype(np.float32)
        data = data.reshape(n_images, n_row, n_col)
        return data


def extract_label(file: str):
    """Extract the label from the idx file"""
    file = DATA_DIR.joinpath(file)
    with open(file, "rb") as stream:
        stream.read(4) # header
        n_labels = int.from_bytes(stream.read(4), byteorder="big")
        buf = stream.read(n_labels)
        data = np.frombuffer(buf, np.uint8)
        data = data.reshape(n_labels, 1)
        return data


class MNIST:

    @staticmethod
    def get_all_train(flatten: bool=True):
        imgs, labels = extract_image("train-images.idx3-ubyte"), extract_label("train-labels.idx1-ubyte")
        imgs = imgs / 256
        if flatten:
            imgs, labels = imgs.reshape(imgs.shape[0], -1), labels.reshape(labels.shape[0], -1)
        return imgs, labels

    @staticmethod
    def get_all_test(flatten: bool=True):
        imgs, labels = extract_image("t10k-images.idx3-ubyte"), extract_label("t10k-labels.idx1-ubyte")
        imgs = imgs / 256
        if flatten:
            imgs, labels = imgs.reshape(imgs.shape[0], -1), labels.reshape(labels.shape[0], -1)
        return imgs, labels
