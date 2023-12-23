import os
from argparse import ArgumentParser
from configs.paths_config import model_paths

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', default=os.path.join(os.path.dirname(__file__), '../exp'), type=str, help='Path to experiment output directory')
		self.parser.add_argument('--device', default='cuda:0', type=str, help='CUDA device to train the model on')
		self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
		self.parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')

		self.parser.add_argument('--lr_D', default=0.00001, type=float, help='Optimizer learning rate for discriminator')
		self.parser.add_argument('--learning_rate_encoder', default=0.00002, type=float, help='Optimizer learning rate for encoder')
		self.parser.add_argument('--learning_rate_renderer', default=0.00001, type=float, help='Optimizer learning rate for renderer')
		self.parser.add_argument('--optim_name_encoder', default='adamw', type=str, help='Which optimizer to use for encoder')
		self.parser.add_argument('--optim_name_renderer', default='adamw', type=str, help='Which optimizer to use for renderer')

		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--l1_lambda', default=1.0, type=float, help='L1 loss multiplier factor')
		self.parser.add_argument('--tri_lambda', default=0.1, type=float, help='Triplanar loss multiplier factor')
		self.parser.add_argument('--depth_lambda', default=1.0, type=float, help='Depth loss multiplier factor')
		self.parser.add_argument('--feature_lambda', default=1.0, type=float, help='Feature loss multiplier factor')
		self.parser.add_argument('--adv_lambda', default=0.025, type=float, help='Adversarial loss multiplier factor')
		
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint to continue training')

		self.parser.add_argument('--max_steps', default=200000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=1000, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--print_interval', default=1, type=int, help='Print interval')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')
		self.parser.add_argument('--add_gan_loss_step', default=0, type=int, help='GAN loss step to the loss function')
		self.parser.add_argument('--use_gan_loss', default=True, type=bool, help='Apply GAN loss to the loss function')
		self.parser.add_argument('--apply_mul_loss_step', default=0, type=int, help='Mul loss Step to the loss function')
		self.parser.add_argument('--use_mul_loss', default=True, type=bool, help='Apply mul loss to the loss function')
		self.parser.add_argument('--use_encoder_scheduler', default=False, type=bool, help='Whether to use scheduler for encoder')
		self.parser.add_argument('--use_renderer_scheduler', default=False, type=bool, help='Whether to use scheduler for renderer')

		# arguments for weights & biases support
		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts
