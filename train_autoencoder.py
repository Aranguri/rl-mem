from autoencoder import AutoEncoder
import argparse
from utils import train_mnist

parser = argparse.ArgumentParser()
parser.add_argument('--mnist', action='store_true', help='train on mnist images?')
parser.add_argument('--run-name', type=str, default=None, help='the name of the run')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.mnist:
        shape = (28,28,1)
        func =train_mnist
    else:
        raise NotImplementedError

    network = AutoEncoder(shape)
    train_mnist(network, 10000, run_name=args.run_name, learning_rate=1.e-3)
