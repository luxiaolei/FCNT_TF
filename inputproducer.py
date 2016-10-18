
import os
import skimage

import numpy as np

from scipy.misc import imread, imresize

from utils import gauss2d


class InputProducer:
	def __init__(self, imgs_path, gt_path, live=False):
		"""

		"""
		self.imgs_path_list = [os.path.join(imgs_path, fn) for fn in sorted(os.listdir(imgs_path))]
		self.gts_list = self.gen_gts(gt_path)
		self.gen_img = self.get_image()

		self.roi_params = {
		'roi_size': 224, 
		'roi_scale': 2,
		'l_off': [0,0]
		}

	def get_image(self):
		idx = -1
		for img_path, gt in zip(self.imgs_path_list, self.gts_list):
			img = imread(img_path, mode='RGB')

			assert min(img.shape[:2]) >= 224

			# Gray to color. RES??
			#if len(img.shape) < 3:
			#img = skimage.color.gray2rgb(img)
			assert len(img.shape) == 3

			idx += 1
			if idx == 0: 
				self.first_gt = gt
				self.first_img = img
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

	def extract_roi(self, img, gt):
		"""
		Extract Regigon of Interest 
		"""
		w, h = gt[2:]
		dia = (w**2 + h**2)**0.5
		scale = [dia / w, dia / h]
		r_w_scale = [self.roi_params['roi_scale']*scale[0],
					 self.roi_params['roi_scale']*scale[1]]

		#print(img.max(), 'origin max')
		h, w = img.shape[:2]
		win_w = gt[2]
		win_h = gt[3]
		win_lt_x = gt[0]
		win_lt_y = gt[1]

		# Center location in img
		win_cx = np.round(win_lt_x + win_w / 2 + self.roi_params['l_off'][0])
		win_cy = np.round(win_lt_y + win_h / 2 + self.roi_params['l_off'][1])

		# Scales the width and height for roi 
		roi_w = r_w_scale[0] * win_w
		roi_h = r_w_scale[1] * win_h

		# Center location in roi
		x1 = win_cx - np.round(roi_w / 2)
		y1 = win_cy - np.round(roi_h / 2)
		x2 = win_cx + np.round(roi_w / 2)
		y2 = win_cy + np.round(roi_h / 2)

		# Out of window detection
		clip = min([x1, y1 ,h-y2 , w-x2])
		pad = 0
		if clip<=0:
		    pad = int(abs(clip)+1)
		    print(clip)
		    img = np.lib.pad(img, [pad, pad], mode='constant', constant_values=[0, 0])
		    x1 = x1 + pad
		    x2 = x2 + pad
		    y1 = y1 + pad
		    y2 = y2 + pad

		# Resize bicubicly 
		#print(img[y1-1:y2, x1-1:x2, :].max(), 'before bicubic resize')
		roi =  imresize(img[y1-1:y2, x1-1:x2, :], [self.roi_params['roi_size'], self.roi_params['roi_size']], interp='bicubic')
		#print(roi.max(), 'after bicubic resize')
		preimg = np.zeros(img.shape[:2])
		roi_pos = [x1, y1, x2-x1+1, y2-y1+1]
		#print(roi.max(), 'roi max')
		#roi = roi.astype(np.float32)
		return roi, roi_pos, preimg, pad

	def gen_mask(self, fea_sz):
		"""
		Generates 2D guassian masked convas with shape same as 
		fea_sz. This method should only called on the first frame.

		Args:
			img_sz: input image size.
			fea_sz: feaure size, to be identical to the 
				Output of sel-CNN net.
		Returns:
			convas: fea_sz shape with 1 channel. The central region is an 
				2D gaussian.
		"""
		im_sz = self.first_img.shape
		x, y, w, h = self.first_gt
		convas = np.zeros(im_sz[:2])

		# Generates 2D gaussian mask
		scale = min([w,h]) / 3 # To be consistence with the paper
		mask = gauss2d([h, w], sigma=scale)
		print(mask.max(), 'max of mask')

		# bottom right coordinate
		x2 = x + w - 1
		y2 = y + h - 1

		# Detects wether the location has out of the img or not
		clip = min(x, y, im_sz[0]-y2, im_sz[1]-x2)
		pad = 0
		if clip <= 0:
			pad = abs(clip) + 1
			convas = np.zeros((im_sz[0] + 2*pad, im_sz[1] + 2*pad))
			x += pad
			y += pad
			x2 += pad
			y2 += pad

		# Overwrite central arear of convas with mask;
		convas[y-1:y2, x-1:x2] = mask
		if clip <= 0:
			# Remove pad
			convas = convas[pad:-pad, pad, -pad]

		if len(convas.shape) < 3:
			convas = skimage.color.gray2rgb(convas)
		assert len(convas.shape) == 3

		# Extrac ROI and resize bicubicly
		convas, _, _, _  = self.extract_roi(convas, self.first_gt)
		print(convas.shape)
		convas = imresize(convas[...,0], fea_sz[:2], interp='bicubic')
		print(convas.max(), 'max convas')

		# Swap back, and normalize
		convas = convas / convas.max()
		#convas = np.transpose(convas)


		return convas#[..., np.newaxis]

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





