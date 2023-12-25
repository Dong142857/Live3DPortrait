import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from criteria import id_loss
from criteria.lpips.lpips import LPIPS
from training.ranger import Ranger
from configs.paths_config import model_paths
from models.triencoder import TriEncoder

class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0
		self.device = self.opts.device

		if self.opts.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.opts)

		self.seed_everything(self.opts.seed)
		if opts.resume:
			self.resume_path = opts.resume_path
		# Initialize network
		# self.net = TriEncoder(self.opts).to(self.device)
		self.net = TriEncoder().to(self.device)

		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		self.id_loss = id_loss.IDLoss().to(self.device).eval()
		self.l1loss = nn.L1Loss().to(self.device).eval()

		# Initialize optimizer
		if self.opts.use_encoder_scheduler:
			self.optimizer_encoder, self.scheduler_encoder = self.configure_optimizers_encoder(self.opts.use_encoder_scheduler)
		else:
			self.optimizer_encoder = self.configure_optimizers_encoder(self.opts.use_encoder_scheduler)

		if self.opts.use_renderer_scheduler:
			self.optimizer_renderer, self.scheduler_renderer = self.configure_optimizers_triplane_renderer(self.opts.use_renderer_scheduler)
		else:
			self.optimizer_renderer = self.configure_optimizers_triplane_renderer(self.opts.use_renderer_scheduler)

		self.D_opt = torch.optim.Adam(self.net.D.parameters(), lr=self.opts.lr_D, betas=(0.5, 0.999))

		# Initialize dataset
		# self supervised learning 

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		# Resume training process from checkpoint path
		if self.opts.checkpoint_path is not None:
			ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")

			print("Load encoder optimizer from checkpoint")
			if "encoder_optimizer" in ckpt:
				self.optimizer_encoder.load_state_dict(ckpt["encoder_optimizer"])
			if "renderer_optimizer" in ckpt:
				self.optimizer_renderer.load_state_dict(ckpt["renderer_optimizer"])
			if "step" in ckpt:
				self.global_step = ckpt["step"]
				print(f"Resuming training process from step {self.global_step}")
			if "best_val_loss" in ckpt:
				self.best_val_loss = ckpt["best_val_loss"]
				print(f"Current best val loss: {self.best_val_loss }")

	def seed_everything(self, seed):
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	def gen_rand_pose(self, pitch_range=26, yaw_range=36, cx_range=0.2, cy_range=0.2, fov_range=4.8, mode="yaw"):
		# set range
		# pitch:  +-26, yaw: +-36/+-49
		if mode == "yaw":
			return (torch.rand(1, device=self.device) - 0.5) * (pitch_range / 180 * torch.pi) + torch.pi/2
		elif mode == "pitch":
			return (torch.rand(1, device=self.device) - 0.5) * (yaw_range / 180 * torch.pi) + torch.pi/2
		elif mode == "cx":
			return (torch.rand(1, device=self.device) - 0.5) * cx_range + 0.5
		elif mode == "cy":
			return (torch.rand(1, device=self.device) - 0.5) * cy_range + 0.5
		elif mode == "fov":
			return (torch.rand(1, device=self.device) - 0.5) * fov_range + 18.837

	def load_pretrain_model(self, path):
		# path = model_paths['encoder_render']
		ckpt = torch.load(path, map_location="cpu")
		# for encoder 
		self.net.encoder.load_state_dict(ckpt["encoder_state_dict"])
		self.net.encoder.requires_grad_(True)
		# for renderer 
		self.net.triplane_renderer.load_state_dict(ckpt["renderer_state_dict"])
		self.net.triplane_renderer.requires_grad_(False)

		# for optimizer
		self.optimizer_encoder.load_state_dict(ckpt["encoder_optimizer"])
		self.optimizer_renderer.load_state_dict(ckpt["render_optimizer"])

		# for discriminator
		self.net.D.load_state_dict(ckpt["discriminator_state"])
		self.net.D.requires_grad_(True)


	def validate(self):
		# fid50k test
		# inception_model = fid50k.load_inception_net(parallel=True)

		pass

	def train_encoder(self):
		self.net.train()
		if self.opts.resume:
			self.load_pretrain_model(self.resume_path)
		D = self.net.D
		D.requires_grad_(True)

		self.net.encoder.requires_grad_(True)
		while self.global_step < self.opts.max_steps: 
			self.optimizer_renderer.zero_grad()
			self.optimizer_encoder.zero_grad()
			 
			triplanes, gt_res, camera_param, ws = self.net.sample_triplane(self.opts.batch_size, self.gen_rand_pose(pitch_range=26, mode='pitch'), self.gen_rand_pose(yaw_range=49, mode='yaw'), fov_deg=self.gen_rand_pose(mode="fov"), cx=self.gen_rand_pose(mode="cx"), cy=self.gen_rand_pose(mode="cy"))
			mul_gt_res, mul_camera_params = self.net.render_from_pretrain(self.opts.batch_size, self.gen_rand_pose(pitch_range=26, mode='pitch'), self.gen_rand_pose(yaw_range=36, mode='yaw'), ws=ws, fov_deg=self.gen_rand_pose(mode="fov"), cx=self.gen_rand_pose(mode="cx"), cy=self.gen_rand_pose(mode="cy"))
			gen_triplanes = self.net.encoder(gt_res['image'])
			render_res = self.net.triplane_renderer(gen_triplanes, camera_param)
			# loss
			image_loss = self.l1loss(render_res['image'], gt_res['image'])
			category_loss = self.id_loss(render_res['image'], gt_res['image'], gt_res['image'])[0]
			raw_loss = self.l1loss(render_res['image_raw'], gt_res['image_raw'])
			depth_loss = self.l1loss(render_res['image_depth'], gt_res['image_depth'])
			feature_loss = self.l1loss(render_res['feature_image'], gt_res['feature_image'])
			image_lpips_loss = self.lpips_loss(render_res['image'], gt_res['image'])
			raw_lpips_loss = self.lpips_loss(render_res['image_raw'], gt_res['image_raw'])

			triloss = self.l1loss(gen_triplanes, triplanes) # triplane loss 
			loss = self.opts.tri_lambda * triloss + self.opts.l1_lambda * image_loss + \
					self.opts.l1_lambda * raw_loss + self.opts.lpips_lambda * image_lpips_loss + \
					self.opts.lpips_lambda * raw_lpips_loss + self.opts.depth_lambda * depth_loss + \
					self.opts.feature_lambda * feature_loss + self.opts.id_lambda * category_loss
			
			if self.opts.use_mul_loss or self.global_step > self.opts.apply_mul_loss_step:
				mul_render_res = self.net.triplane_renderer(gen_triplanes, mul_camera_params)
				mul_image_loss = self.l1loss(mul_render_res['image'], mul_gt_res['image'])
				mul_raw_loss = self.l1loss(mul_render_res['image_raw'], mul_gt_res['image_raw'])
				mul_depth_loss = self.l1loss(mul_render_res['image_depth'], mul_gt_res['image_depth'])
				mul_feature_loss = self.l1loss(mul_render_res['feature_image'], mul_gt_res['feature_image'])
				mul_image_lpips_loss = self.lpips_loss(mul_render_res['image'], mul_gt_res['image'])
				mul_raw_lpips_loss = self.lpips_loss(mul_render_res['image_raw'], mul_gt_res['image_raw'])
				mul_category_loss = self.id_loss(mul_render_res['image'], mul_gt_res['image'], mul_gt_res['image'])[0]
				loss += self.opts.l1_lambda * mul_image_loss + self.opts.l1_lambda * mul_raw_loss + \
						self.opts.lpips_lambda * mul_image_lpips_loss + self.opts.lpips_lambda * mul_raw_lpips_loss + \
						self.opts.depth_lambda * mul_depth_loss + self.opts.feature_lambda * mul_feature_loss + \
						self.opts.id_lambda * mul_category_loss

			if self.opts.use_gan_loss or self.global_step > self.opts.add_gan_loss_step:
				# loss += self.opts.gan_lambda * self.net.discriminator_loss(render_res['image'])
				r1_gamma = 10
				self.D_opt.zero_grad()
				gt_img_tmp_image = gt_res['image'].detach().requires_grad_(True)
				gt_img_tmp_raw = gt_res['image_raw'].detach().requires_grad_(True)
				gt_logits = D({'image_raw': gt_img_tmp_raw, 'image': gt_img_tmp_image}, camera_param)
				loss_Dgt = torch.nn.functional.softplus(-gt_logits).mean()

				# r1 regularization
				r1_grads = torch.autograd.grad(outputs=[gt_logits.sum()], inputs=[gt_img_tmp_image, gt_img_tmp_raw], create_graph=True, only_inputs=True)
				r1_grads_image = r1_grads[0]
				r1_grads_image_raw = r1_grads[1]
				r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
				loss_Dr1 = (r1_penalty * (r1_gamma / 2))
				(loss_Dgt + loss_Dr1).mean().backward(retain_graph=True)

				pred_logits = D({'image_raw': render_res['image_raw'].detach(), 'image': render_res['image'].detach()}, camera_param)
				loss_Dgen = torch.nn.functional.softplus(pred_logits).mean()
				loss_Dgen.backward()
				self.D_opt.step()

				# gan loss for generator:
				logits = D({'image_raw': render_res['image_raw'], 'image': render_res['image']}, camera_param)
				loss_gan = torch.nn.functional.softplus(-logits).mean()
				loss += self.opts.adv_lambda * loss_gan
				# loss_gan.backward(retain_graph=True)
							

			loss.backward()
			self.optimizer_renderer.step()
			self.optimizer_encoder.step()
			if self.opts.use_encoder_scheduler:
				self.scheduler_encoder.step(self.global_step)
			if self.opts.use_renderer_scheduler:
				self.scheduler_renderer.step(self.global_step)
			self.global_step += 1

			if self.global_step % self.opts.print_interval == 0:
				print(f"Step {self.global_step}: loss = {loss.item()}, triloss = {triloss.item()}")
				print(f"image_loss = {image_loss.item()}, raw_loss = {raw_loss.item()}")
				print(f"image_lpips_loss = {image_lpips_loss.item()}, raw_lpips_loss = {raw_lpips_loss.item()}")
				print(f"loss_depth = {depth_loss.item()}, loss_feature = {feature_loss.item()}")
				self.logger.add_scalar("loss", loss.item(), self.global_step)
				self.logger.add_scalar("triloss", triloss.item(), self.global_step)
				self.logger.add_scalar("image_loss", image_loss.item(), self.global_step)
				self.logger.add_scalar("raw_loss", raw_loss.item(), self.global_step)
				self.logger.add_scalar("image_lpips_loss", image_lpips_loss.item(), self.global_step)
				self.logger.add_scalar("raw_lpips_loss", raw_lpips_loss.item(), self.global_step)
				self.logger.add_scalar("loss_depth", depth_loss.item(), self.global_step)
				self.logger.add_scalar("loss_feature", feature_loss.item(), self.global_step)
				self.logger.add_scalar("loss_id", category_loss.item(), self.global_step)
				# self.logger.add_scalar("encoder_learning_rate", self.optimizer_encoder.param_groups[0]['lr'], self.global_step)
				# self.logger.add_scalar("renderer_learning_rate", self.optimizer_renderer.param_groups[0]['lr'], self.global_step)

				if self.opts.use_gan_loss or self.global_step > self.opts.add_gan_loss_step:
					print(f"loss_gan = {loss_gan.item()}, loss_Dgt = {loss_Dgt.item()}, loss_Dgen = {loss_Dgen.item()}")
					self.logger.add_scalar("loss_gan", loss_gan.item(), self.global_step)
					self.logger.add_scalar("loss_Dgt", loss_Dgt.item(), self.global_step)
					self.logger.add_scalar("loss_Dgen", loss_Dgen.item(), self.global_step)
				if self.opts.use_mul_loss or self.global_step > self.opts.apply_mul_loss_step:
					print(f"mul_image_loss = {mul_image_loss.item()}, mul_raw_loss = {mul_raw_loss.item()}")
					print(f"mul_image_lpips_loss = {mul_image_lpips_loss.item()}, mul_raw_lpips_loss = {mul_raw_lpips_loss.item()}")
					print(f"mul_loss_depth = {mul_depth_loss.item()}, mul_loss_feature = {mul_feature_loss.item()}")
					self.logger.add_scalar("mul_image_loss", mul_image_loss.item(), self.global_step)
					self.logger.add_scalar("mul_raw_loss", mul_raw_loss.item(), self.global_step)
					self.logger.add_scalar("mul_image_lpips_loss", mul_image_lpips_loss.item(), self.global_step)
					self.logger.add_scalar("mul_raw_lpips_loss", mul_raw_lpips_loss.item(), self.global_step)
					self.logger.add_scalar("mul_loss_depth", mul_depth_loss.item(), self.global_step)
					self.logger.add_scalar("mul_loss_feature", mul_feature_loss.item(), self.global_step)
					self.logger.add_scalar("mul_loss_id", mul_category_loss.item(), self.global_step)

			if self.global_step % self.opts.image_interval == 0:
				# Image.fromarray(((1 + render_res['image_raw'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255).astype(np.uint8)).save('debug_image8/' + 'test_raw' + str(self.global_step) + '.png')
				# Image.fromarray(((1 + render_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255).astype(np.uint8)).save('debug_image/' + 'test' + str(self.global_step) + '.png')
				# Image.fromarray(((1 + gt_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255).astype(np.uint8)).save('debug_image/' + 'testgt' + str(self.global_step) + '.png')
				vis_render_res = (1 + render_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255
				vis_gt_res = (1 + gt_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255
				plt.imsave("debug_image/" + "test_vis" + str(self.global_step) + ".png", np.concatenate([vis_render_res, vis_gt_res], axis=1).astype(np.uint8))

			if self.global_step % self.opts.save_interval == 0:
				self.checkpoint_me(is_best=False)

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps: 
			self.optimizer_renderer.zero_grad()
			triplanes, gt_res, camera_param = self.net.sample_triplane(self.opts.batch_size, self.gen_rand_pose(), self.gen_rand_pose())
			render_res = self.net.triplane_renderer(triplanes, camera_param)
			# loss
			loss = self.opts.l2_lambda * self.mse_loss(render_res['image'], gt_res['image'])
			loss += self.opts.l2_lambda * self.mse_loss(render_res['image_raw'], gt_res['image_raw'])
			loss += self.opts.lpips_lambda * self.lpips_loss(render_res['image'], gt_res['image'])
			loss += self.opts.lpips_lambda * self.lpips_loss(render_res['image_raw'], gt_res['image_raw'])
			# loss += self.opts.id_lambda * self.id_loss(render_res['image'], gt_res['image'], gt_res['image'])
			# loss += self.opts.id_lambda * self.id_loss(render_res['image_raw'], gt_res['image_raw'], gt_res['image_raw'])

			loss.backward()
			self.optimizer_renderer.step()
			self.global_step += 1

			if self.global_step % self.opts.print_interval == 0:
				print(f"Step {self.global_step}: loss = {loss.item()}")
				self.logger.add_scalar("loss", loss.item(), self.global_step)

			if self.global_step % self.opts.image_interval == 0:
				Image.fromarray(((1 + render_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255).astype(np.uint8)).save('debug_image11/' + 'test' + str(self.global_step) + '.png')
				
			if self.global_step % self.opts.save_interval == 0:
				self.checkpoint_me(is_best=False)

	def checkpoint_me(self, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)

	def configure_optimizers_encoder(self, add_scheduler=False):
		params = list(self.net.encoder.parameters())
		if self.opts.optim_name_encoder == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_encoder)
		elif self.opts.optim_name_encoder == 'adamw':
			optimizer = torch.optim.AdamW(params, lr=self.opts.learning_rate_encoder)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate_encoder)

		if add_scheduler:
			# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opts.max_steps/20) # 5个周期
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 
			return optimizer, scheduler
		else:
			return optimizer

	def configure_optimizers_triplane_renderer(self, add_scheduler=False):
		params = list(self.net.triplane_renderer.parameters())
		if self.opts.optim_name_renderer == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_renderer)
		elif self.opts.optim_name_encoder == 'adamw':
			optimizer = torch.optim.AdamW(params, lr=self.opts.learning_rate_renderer)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate_renderer)

		if add_scheduler:
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=200)
			return optimizer, scheduler
		else:
			return optimizer


	def __get_save_dict(self):
		save_dict = {
			'encoder_state_dict': self.net.encoder.state_dict(),
			'renderer_state_dict': self.net.triplane_renderer.state_dict(),
			'discriminator_state': self.net.D.state_dict(),
			'opts': vars(self.opts),
			'best_val_loss': self.best_val_loss,
			'step': self.global_step,
			'encoder_optimizer': self.optimizer_encoder.state_dict(),
			'render_optimizer': self.optimizer_renderer.state_dict(),
					}
		return save_dict
