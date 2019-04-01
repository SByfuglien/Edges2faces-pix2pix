"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
	min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
	<modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
	<__init__>: Initialize this model class.
	<set_input>: Unpack input data and perform data pre-processing.
	<forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
	<optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks


class TemplateModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""Add new model-specific options and rewrite default values for existing options.

		Parameters:
			parser -- the option parser
			is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""
		parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='edges2faces')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
		if is_train:
			parser.set_defaults(pool_size=0, gan_mode='vanilla')
			parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

		return parser

	def __init__(self, opt):
		"""Initialize this model class.

		Parameters:
			opt -- training/test options

		A few things can be done here.
		- (required) call the initialization function of BaseModel
		- define loss function, visualization images, model names, and optimizers
		"""
		BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
		# specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
		self.loss_names = ['G_L1', 'G_GAN', 'D_real', 'D_fake ']
		# specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
		self.visual_names = ['real', 'fake', 'result']
		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
		# you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
		if self.isTrain:
			self.model_names = ['G', 'D']
		else:  # during test time, only load G
			self.model_names = ['G']
		# define networks; you can use opt.isTrain to specify different behaviors for training and test.
		self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)

		if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
			self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
			                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

		if self.isTrain:  # only defined during training time
			# define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
			# We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
			self.criterionL1 = torch.nn.L1Loss()
			self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
			# define and initialize optimizers. You can define one optimizer for each network.
			# If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

		# Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

	def set_input(self, input):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input: a dictionary that contains the data itself and its metadata information.
		"""
		AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
		self.edges = input['A' if AtoB else 'B'].to(self.device)  # get image data A
		self.faces = input['B' if AtoB else 'A'].to(self.device)  # get image data B
		self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

	def forward(self):
		"""Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
		self.result = self.netG(self.edges)  # generate output image given the input data_A

	def backward_D(self):
		"""Calculate losses, gradients, and update network weights; called in every training iteration"""
		# Fake; stop backprop to the generator by detaching fake_B
		fake = torch.cat((self.edges, self.result),
		                    1)  # we use conditional GANs; we need to feed both input and output to the discriminator
		pred_fake = self.netD(fake.detach())
		self.loss_D_fake = self.criterionGAN(pred_fake, False)
		# Real
		real = torch.cat((self.edges, self.faces), 1)
		pred_real = self.netD(real)
		self.loss_D_real = self.criterionGAN(pred_real, True)
		# combine loss and calculate gradients
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		self.loss_D.backward()

	def backward_G(self):
		# First, G(A) should fake the discriminator
		fake = torch.cat((self.edges, self.result), 1)
		pred_fake = self.netD(fake)
		self.loss_G_GAN = self.criterionGAN(pred_fake, True)
		# Second, G(A) = B
		self.loss_G_L1 = self.criterionL1(self.result, self.faces) * self.opt.lambda_L1
		# combine loss and calculate gradients
		self.loss_G = self.loss_G_GAN + self.loss_G_L1
		self.loss_G.backward()

	def optimize_parameters(self):
		"""Update network weights; it will be called in every training iteration."""
		self.forward()               # first call forward to calculate intermediate results
		# Update D
		self.set_requires_grad(self.netD, True) # enable backprop for D
		self.optimizer_D.zero_grad()   # clear network G's existing gradients
		self.backward_D()              # calculate gradients for network D
		self.optimizer_D.step()        # update gradients for network D
		# Update g
		self.set_requires_grad(self.netD, False)
		self.optimizer_G.zero_grad()  # clear network G's existing gradients
		self.backward_G()  # calculate gradients for network G
		self.optimizer_G.step()  # update gradients for network G
