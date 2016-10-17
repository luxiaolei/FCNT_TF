

import numpy as np

import os
import skimage


class InputProducer:
	def __init__(self, imgs_path, gt_path, live=False):
		"""

		"""
		self.imgs_path_list = [os.path.join(imgs_path, fn) for fn in sorted(os.listdir(imgs_path))]
		self.gts_list = self.gen_gts(gt_path)
		self.img_generator = self.get_image()

	def get_image(self):
		idx = -1
		for img_path, gt in zip(self.imgs_path_list, self.gts_list):
			img = skimage.io.imread(img_path)
			assert min(img.shape[:2]) >= 224
			idx += 1
			yield img, gt, idx


	def gen_gts(self, gt_path):
		"""
		Each row in the ground-truth files represents the bounding box 
		of the target in that frame. (tl_x, tl_y, box-width, box-height)
		"""
		f = open(gt_path, 'r')
		lines = f.readlines()

		try:
			gts_list = [[int(p) for p in i[:-1].split(',')] 
			                   for i in lines]
		except Exception as e:
			gts_list = [[int(p) for p in i[:-1].split('\t')] 
			                   for i in lines]
		return gts_list

	# Deprecated method.
	def porcess_img(img):
		"""
		Porcessing image required by vgg16
		Returns:
		image of shape [224, 224, 3]
		[1, height, width, depth]
		"""
		# load image
		img = img / 255.0
		assert (0 <= img).all() and (img <= 1.0).all()

		# conert to color image if its a grey one
		if len(img.shape) < 3:
			img = skimage.color.gray2rgb(img)
		assert len(img.shape) == 3

		# crop image from center
		short_edge = min(img.shape[:2])
		yy = int((img.shape[0] - short_edge) / 2)
		xx = int((img.shape[1] - short_edge) / 2)
		crop_img = img[yy : yy + short_edge, xx : xx + short_edge]

		# resize to 224, 224
		resized_img = skimage.transform.resize(crop_img, (224, 224))
		return resized_img.reshape((1, 224, 224, 3))

	# Deprecated method.
	def img_porcess(img):
		img = img.astype(float)
		# conert to color image if its a grey one
		if len(img.shape) < 3:
			img = skimage.color.gray2rgb(img)

		# Swap x,y order and subtract mean value
		mean_pix = [123.68, 116.779, 103.939] # BGR
		img = np.transpose(img, [1,0,2])
		img[:, :, 0] -= mean_pix[0]
		img[:, :, 1] -= mean_pix[1]
		img[:, :, 2] -= mean_pix[2]
		return img.reshape((1, 224, 224, 3))





