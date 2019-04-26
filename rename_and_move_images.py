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


root_dir = "C:/Users/Sivert/Datasets/dataset-5k/faces/"
destination_dir = "C:/Users/Sivert/Datasets/dataset-5k/edges/"
# root_dir = "dataset-2k/train/"
# destination_dir = "dataset-2k/val/"

# numbers = []
# for root, dirs, files in os.walk(root_dir, topdown=True):
# 	for file in sorted(files, key=natural_keys):
# 		string1 = file.split("/")[-1]
# 		number = file.split("_B")[0]
# 		numbers.append(number)
# print(numbers)
picture_num = 1
for root, dirs, files in os.walk(root_dir, topdown=True):
	for file in sorted(files, key=natural_keys):
		# if str(picture_num) not in numbers:
		# 	print(picture_num)
		# 	# os.rename(os.path.join(root, file), destination_dir + str(picture_num) + "_A.jpg")
		# 	os.remove(os.path.join(root, file))
		# 	picture_num += 1
		# 	continue
		os.rename(os.path.join(root, file), root_dir + str(picture_num) + "_B.jpg")
		# os.remove(os.path.join(root, file))
		picture_num += 1


