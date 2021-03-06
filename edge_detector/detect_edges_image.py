# USAGE
# python detect_edges_image.py --edge-detector hed_model --image images/guitar.jpg

# import the necessary packages
import argparse
import re

import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge-detector", type=str, required=True,
				help="path to OpenCV's deep learning edge detector")
ap.add_argument("-i", "--image-dir", type=str, required=True,
				help="path to input image directory")
args = vars(ap.parse_args())


class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]


# load our serialized edge detector from disk
print("[INFO] loading edge detector...")
protoPath = os.path.sep.join([args["edge_detector"],
							  "deploy.prototxt"])
modelPath = os.path.sep.join([args["edge_detector"],
							  "hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)


def atoi(text):
	return int(text) if text.isdigit() else text


def natural_keys(text):
	'''
	alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)
	'''
	return [atoi(c) for c in re.split(r'(\d+)', text)]


root_dir = args["image_dir"]
destination_dir = "C:/Users/Sivert/git/Edges2faces-pix2pix/drawing/"
picture_num = 1
for root, dirs, files in os.walk(root_dir, topdown=True):
	for file in sorted(files, key=natural_keys):

		image = cv2.imread(os.path.join(root, file))
		(H, W) = image.shape[:2]

		# construct a blob out of the input image for the Holistically-Nested
		# Edge Detector
		blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
									 mean=(104.00698793, 116.66876762, 122.67891434),
									 swapRB=False, crop=False)

		# set the blob as the input to the network and perform a forward pass
		# to compute the edges
		print("[INFO] performing holistically-nested edge detection on " + file)
		net.setInput(blob)
		hed = net.forward()
		hed = cv2.resize(hed[0, 0], (W, H))
		hed = (255 * hed).astype("uint8")

		# show the output edge detection results for Canny and
		# Holistically-Nested Edge Detection
		hed = cv2.bitwise_not(hed)
		cv2.imwrite(destination_dir + str(picture_num) + "_A" + ".jpg", hed)
		picture_num += 1
