"""Standalone JiT deraining training.

Protocol:
  - Train on RainTrainL.
  - Validate on the complete Rain100L test set.

The dehazing entry point remains in ``train_restoration.py``. Both tasks share
the same rectified-flow model and training loop so their checkpoints can later
be compared before building a joint multi-task model.

Example:
  python train_deraining.py \
      --data_file_dir ./datasets \
      --output_dir ./output/jit_rain_rectified \
      --model JiT-B/16 \
      --img_size 512 --patch_size 512 \
      --batch_size 2 --epochs 50 --lr 1e-4
"""

from pathlib import Path

from data.dataset_utils import RainDerainDataset
from train_restoration import get_args_parser as get_base_parser
from train_restoration import run_training


def get_args_parser():
    parser = get_base_parser()
    parser.description = "JiT deraining (RainTrainL train / Rain100L val)"
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
        help="Random-crop repetitions per RainTrainL pair each epoch",
    )
    parser.set_defaults(output_dir="./output/jit_rain_rectified")
    return parser


def main(args):
    dataset_train = RainDerainDataset(args, split="train")
    dataset_val = RainDerainDataset(args, split="test")
    run_training(args, dataset_train, dataset_val)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
