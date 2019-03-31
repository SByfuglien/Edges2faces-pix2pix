import os
import glob

root_dir = "C:/Users/Sivert/Datasets/CyberextruderUltimate/**"
destination_dir = "C:/Users/Sivert/Datasets/face-matching-dataset/targets/"
picture_num = 1
for filename in glob.iglob(root_dir, recursive=True):
	if os.path.isfile(filename):
		os.rename(filename, destination_dir + "target_" + str(picture_num) + ".jpg")
		picture_num += 1
