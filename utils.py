
import numpy as np
import skimage


# returns the top1 string
def print_prob(prob):
	synset = [l.strip() for l in open('synset.txt').readlines()]
	#print prob
	print "prob shape", prob.shape
	pred = np.argsort(prob)[::-1]

	# Get top1 label
	top1 = synset[pred[0]]
	print "Top1: ", top1
	# Get top5 label
	top5 = [synset[pred[i]] for i in range(5)]
	print "Top5: ", top5
	return top1


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


def img_with_bbox(img_origin, gt):
	"""Returns Image with bounding box."""
    w, h = gt[2:]
    tl_x, tl_y = gt[:2]
    tr_x, tr_y = tl_x + w, tl_y 
    dl_x, dl_y = tl_x, tl_y + h
    dr_x, dr_y = tl_x + w, tl_y +h

    rr1, cc1 = skimage.draw.line( tl_y,tl_x, tr_y, tr_x)
    rr2, cc2 = skimage.draw.line( tl_y,tl_x, dl_y, dl_x)
    rr3, cc3 = skimage.draw.line( dr_y,dr_x, tr_y, tr_x)
    rr4, cc4 = skimage.draw.line( dr_y,dr_x, dl_y, dl_x)

    img_origin[rr1, cc1] = 1
    img_origin[rr2, cc2] = 1
    img_origin[rr3, cc3] = 1
    img_origin[rr4, cc4] = 1
    return img_origin


