# coding=utf-8
# 导入相应的python包
import argparse
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import matplotlib.image as mpimg
from scipy import misc
import cv2




# 读取图片并将其转化为浮点型
# image = cv2.imread('test1.png')
# image = img_as_float(io.imread(args["image"]))
# img_slic = np.zeros((image.shape[0],image.shape[1]),dtype=np.float)
# # 循环设置不同的超像素组
# seg_scale = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,100]
# for numSegments in (100, 200, 300):
# 	# 应用slic算法并获取分割结果
# 	segments = slic(image, n_segments = numSegments, sigma = 5)
    # img_slic = np.concatenate([img_slic,segments],axis=0)
# img_slic = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float)
##########################图像替换的slic######################################
def slic_seg(image,seg_scale):
	# image_int = image.astype(np.int32)
	# seg_num = len(seg_scale)
	image_slic = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float)
	for numSegments in seg_scale:
		# 应用slic算法并获取分割结果
		segments = slic(image, n_segments=numSegments, sigma=5)
		segments_val = np.unique(segments)
		for i in segments_val:
			 image[segments == i] = np.mean(image[segments == i])
		###################
		# fig = plt.figure("Superpixels -- %d segments" % (numSegments))
		# ax = fig.add_subplot(1, 1, 1)
		# ax.imshow(mark_boundaries(image, segments))
		# plt.axis("off")
		# plt.show()
		# img_slic[seg_num] = segments
		####################
		image_new = image[:,:,0][np.newaxis,:]
		image_slic = np.concatenate([image_slic, image_new])
	image_slic = image_slic[1:]
	image_slic = image_slic.transpose([1,2,0])
	return image_slic


#################备份##########################
# def slic_seg(image,seg_scale):
# 	# seg_num = len(seg_scale)
# 	img_slic = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float)
# 	for numSegments in seg_scale:
# 		# 应用slic算法并获取分割结果
# 		segments = slic(image, n_segments=numSegments, sigma=5)
# 		fig = plt.figure("Superpixels -- %d segments" % (numSegments))
# 		ax = fig.add_subplot(1, 1, 1)
# 		ax.imshow(mark_boundaries(image, segments))
# 		plt.axis("off")
# 		plt.show()
# 		# img_slic[seg_num] = segments
# 		segments = segments[np.newaxis,:]
# 		img_slic = np.concatenate([img_slic, segments],axis=0)
# 	img_slic = img_slic[1:]
# 	return img_slic

# if __name__ == '__main__':
	# image = cv2.imread('test1.png')
	# img_slic = np.zeros((image.shape[0], image.shape[1]), dtype=np.float)
	# 循环设置不同的超像素组
	# seg_scale = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # img_slic = slic_seg(image,seg_scale)




