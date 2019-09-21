import argparse
from ..basic_params import basic_parser

parser = argparse.ArgumentParser(parents=[basic_parser], conflict_handler="resolve")


parser.add_argument("-cw", action="store_true", dest="cls_wise_accu")

parser.add_argument("--steps", type=int, default=10000)

parser.add_argument("--lr", type=float, default=0.01)

parser.add_argument("--batch_size", type=int, default=36)

parser.add_argument("--eval_batch_size", type=int, default=32)

# parser.add_argument("--eval_per_step", type=int, default=10)

params = parser.parse_args()

