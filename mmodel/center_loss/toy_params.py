import argparse
from ..basic_params import basic_parser

parser = argparse.ArgumentParser(parents=[basic_parser], conflict_handler="resolve")


parser.add_argument("--steps", type=int, default=20000)

parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--eval_batch_size", type=int, default=64)

params = parser.parse_args()

