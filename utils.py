






def extract_roi(image,location, scale):
	"""
	Extract Regigon of Interest 
	"""
	pass


def gen_mask():
	"""
	
	"""

	# Construct a 2D gaussian mask with variance proportional to 
	# the target scale size


	# Extract ROI from the Gaussian kernel

	pass


def IOU_eval(groud_truth_box, predicted_box):
	"""
	Returns:
		iou: scaler
	"""

	pass


def draw_on_img(img, gt, pre, IoU_score):
	"""
	Draw ground truth bbox and predicted bbox on given image, along
	with its IoU score.
	"""
	pass


