"""
This file runs the main training/val loop
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
	opts = TrainOptions().parse()
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)
	# if not os.path.exists(opts.exp_dir):
	# 	os.mkdir(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	# coach.train()
	coach.train_encoder()


if __name__ == '__main__':
	main()
