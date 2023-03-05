from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--train-path', dest='train_path')
parser.add_argument('--test-path', dest='test_path')
parser.add_argument('--batch-size', dest='batch_size', default=16, type=int)
parser.add_argument('--gamma', dest='gamma', default=0.5)
parser.add_argument('--no-epoch', dest='no_epoch', default=10, type=int)
parser.add_argument('--learning-rate', dest='lr', default=0.05, type=float)
parser.add_argument('--use-cuda', dest='use_cuda', default=True)

args = parser.parse_args()