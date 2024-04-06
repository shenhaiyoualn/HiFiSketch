"""

CUDA_VISIBLE_DEVICES="1" python scripts/train.py \
--dataset_type=CUHK \
--encoder_type=hifinet \
--exp=experiments/CUHK \
--workers=1 \
--batch_size=4 \
--test_batch_size=2 \
--test_workers=1 \
--val_interval=5000 \
--save_interval=5000 \
--n_iters_per_batch=1 \
--max_val_batches=150 \
--output_size=1024 \
--load_w_encoder
"""

import json
import os
import sys
import pprint
import warnings
warnings.filterwarnings("ignore")

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_hifisketch import Coach


def main():
	opts = TrainOptions().parse()
	create_initial_experiment_dir(opts)
	coach = Coach(opts)
	coach.train()


def create_initial_experiment_dir(opts):
	os.makedirs(opts.exp, exist_ok=True)
	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
	main()
