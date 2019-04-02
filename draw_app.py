import cv2
import numpy as np
from options.test_options import TestOptions
from models import create_model
import torch
from data.base_dataset import get_transform, get_params
from PIL import Image
from data import create_dataset
from util import util
from scipy.misc import imresize

opt = TestOptions().parse()
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
opt.load_iter = 0
opt.epoch = 'latest'
model = create_model(opt)  # create a model given opt.model and other options
model.setup(opt)

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
running = True
paint_size = 5


def forward_pass():
	dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
	if opt.eval:
		model.eval()
	# print(len(dataset))
	for i, data in enumerate(dataset):
		if i >= opt.num_test:  # only apply our model to opt.num_test images.
			break
		model.set_input(data)  # unpack data from data loader
		model.test()  # run inference
		visuals = model.get_current_visuals()  # get image results
		kas, image = visuals.popitem()
		im = util.tensor2im(image)
		h, w, _ = im.shape
		imag = Image.fromarray(im)
		cv2.imshow("generated", im)
		imag.save("results/img.jpg")


# print("done")


# mouse callback function
def draw_circle(event, x, y, flags, param):
	global ix, iy, drawing, running, paint_size

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix, iy = x, y

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing:
			if flags == 9:
				cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 20)
				ix, iy = x, y
			else:
				cv2.line(img, (ix, iy), (x, y), (0, 0, 0), paint_size)
				ix, iy = x, y
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		cv2.imwrite("drawing/draw.jpg", img)
		forward_pass()
	# elif event == cv2.EVENT_RBUTTONDOWN:
	# 	cv2.imwrite("drawing/draw.jpg", img)
	# 	forward_pass()
	elif event == cv2.EVENT_MOUSEWHEEL:
		if flags > 0 and paint_size < 10:
			paint_size += 1
		elif flags < 0 and paint_size > 1:
			paint_size -= 1
	elif event == cv2.EVENT_MBUTTONUP:
		img.fill(255)
	elif event == cv2.EVENT_MBUTTONDBLCLK:
		running = False
		cv2.destroyAllWindows()


img = np.zeros((600, 600, 3), np.uint8)
img.fill(255)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while running:
	cv2.imshow('image', img)
	if cv2.waitKey(1) == 27:
		running = False
