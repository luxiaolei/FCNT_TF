

import skimage
import skimage.io
import skimage.transform
import numpy as np




def load_image(path):




class InputProducer:
	def __init__(self, imgs_path, gt_path, live=False):
		"""

		"""
		# conert Imgs_path into a list of img path
		self.img_fns = []

		# read gt_path into a list
		self.gts = []




	def get_image(self):
		idx = 0
		for img_path, gt in zip((self.img_fns, self.gts)):
			img = self.porcess_img(img_path)

			# convert gt
			gt = self.porcess_gt(gt)
			idx += 1
			yield img, idx, gt

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


	def porcess_gt(self, gt):
		"""
		Each row in the ground-truth files represents the bounding box 
		of the target in that frame. (tl_x, tl_y, box-width, box-height)
		"""
		return gt




	def 


def static_imgs(path):
	"""
	Generater 

	Returns:
		img: 
		index:
		bbox: 
	"""
	pass


def live_video(stream):
	"""
	Returns:

	"""
	pass
