import os
import re


def atoi(text):
	return int(text) if text.isdigit() else text


def natural_keys(text):
	'''
	alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)
	'''
	return [atoi(c) for c in re.split(r'(\d+)', text)]


root_dir = "C:/Users/Sivert/Datasets/face-matching-dataset/targets/"
destination_dir = "C:/Users/Sivert/Datasets/face-matching-dataset/targets/"
# root_dir = "dataset/faces/train/**"
# destination_dir = "dataset/faces/train/"
picture_num = 1
for root, dirs, files in os.walk(root_dir, topdown=True):
	for file in sorted(files, key=natural_keys):
		print("{}/{}".format(picture_num, len(files)))
		# if picture_num < 492:
		# 	picture_num += 1
		# 	continue
		os.rename(os.path.join(root, file), destination_dir + str(picture_num) + "_B" + ".jpg")
		picture_num += 1

