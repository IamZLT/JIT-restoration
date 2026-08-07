"""Dynamic-resolution JiT deraining training.

Train: RainTrainL multi-scale rectangular crops.
Validation: Rain100L native-resolution pairs.
"""

from pathlib import Path

from data.dataset_dynamic import DynamicRainDataset
from train_restoration_dynamic import (
    get_args_parser as get_dynamic_parser,
)
from train_restoration_dynamic import run_dynamic_training


def get_args_parser():
    parser = get_dynamic_parser()
    parser.description = "Dynamic-resolution JiT deraining"
    parser.add_argument(
        "--rain_train_root",
        default=None,
        type=str,
        help="Optional RainTrainL root containing rainy/ and gt/",
    )
    parser.add_argument(
        "--rain_test_root",
        default=None,
        type=str,
        help="Optional Rain100L root containing rainy/ and gt/",
    )
    parser.add_argument(
        "--rain_repeat",
        default=10,
        type=int,
        help="Number of crop samples per RainTrainL pair and epoch",
    )
    parser.set_defaults(output_dir="./output/jit_rain_dynamic")
    return parser


def main(args):
    dataset_train = DynamicRainDataset(args, split="train")
    dataset_val = DynamicRainDataset(args, split="test")
    run_dynamic_training(args, dataset_train, dataset_val)


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
