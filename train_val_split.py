import os
import re
import random


def atoi(text):
	return int(text) if text.isdigit() else text


def natural_keys(text):
	'''
	alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)
	'''
	return [atoi(c) for c in re.split(r'(\d+)', text)]


root_dir = "dataset-full/train/"
destination_dir = "dataset-full/val/"
picture_num = 1461
for root, dirs, files in os.walk(root_dir, topdown=True):
	for file in sorted(files, key=natural_keys):
		if random.random() < 0.05:
			os.rename(os.path.join(root, file), destination_dir + file)
			picture_num += 1
			print(picture_num)
			if picture_num == 1538:
				break
