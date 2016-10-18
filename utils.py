
import numpy as np
import tensorflow as tf
import skimage

from scipy.misc import imresize


def variable_on_cpu(scope, name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	  name: name of the variable
	  shape: list of ints
	  initializer: initializer for Variable

	Returns:
	  Variable Tensora

	"""
	dtype = tf.float32
	with tf.variable_scope(scope) as scope:
		with tf.device('/cpu:0'):
			variable = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return variable

def variable_with_weight_decay(scope, name, shape, stddev=1e-3, wd=None):
	"""Helper to create an initialized Variable with weight decay

	Args:
		name: name of the variable
		shape: list of ints
		stddev: float, standard deviation of a truncated Gaussian for initial value
		wd: add L2loss weight decay multiplied by this float. If None, weight decay 
				is not added to this variable

	Returns:
		Variable: Tensor
	"""
	dtype = tf.float32
	variable = variable_on_cpu(
							scope,
							name, 
							shape,
							initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(variable), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return variable


# returns the top1 string
def print_prob(prob):
	synset = [l.strip() for l in open('synset.txt').readlines()]
	#print prob
	print( "prob shape", prob.shape)
	pred = np.argsort(prob)[::-1]

	# Get top1 label
	top1 = synset[pred[0]]
	print( "Top1: ", top1)
	# Get top5 label
	top5 = [synset[pred[i]] for i in range(5)]
	print( "Top5: ", top5)
	return top1

# draw on img
def img_with_bbox(img_origin, gt_1):
	img =np.copy(img_origin)
	gt_1 = [int(i) for i in gt_1]
	w, h = gt_1[2:]
	tl_x, tl_y = gt_1[:2]
	tr_x, tr_y = tl_x + w, tl_y 
	dl_x, dl_y = tl_x, tl_y + h
	dr_x, dr_y = tl_x + w, tl_y +h

	rr1, cc1 = skimage.draw.line( tl_y,tl_x, tr_y, tr_x)
	rr2, cc2 = skimage.draw.line( tl_y,tl_x, dl_y, dl_x)
	rr3, cc3 = skimage.draw.line( dr_y,dr_x, tr_y, tr_x)
	rr4, cc4 = skimage.draw.line( dr_y,dr_x, dl_y, dl_x)
	img[rr1, cc1, :] = 1
	img[rr2, cc2, :] = 1
	img[rr3, cc3, :] = 1
	img[rr4, cc4, :] = 1
	return img

def gauss2d(shape=(6,6),sigma=0.5):
	"""
	2D gaussian mask - should give the same result as MATLAB's
	fspecial('gaussian',[shape],[sigma])
	"""
	# Implements 2D gaussian formula
	m,n = [(ss-1.)/2. for ss in shape]
	y,x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh

	# Normalize
	#h = h / h.max()
	return h

def extract_roi_old(img, gt, roi_size):
	"""Extract ROI from img with target centered.

	Returns:
	    roi: tensor,
	    roi_pos: list of params for roi_pos, [tlx, tly, h, w]

	"""
	assert max(gt[2:]) <= roi_size

	# Construct an padded img first.
	convas = np.zeros([img.shape[0]+2*roi_size, img.shape[1]+2*roi_size, 3])
	convas[roi_size:-roi_size, roi_size:-roi_size] = img

	# Compute target center location in convas
	tlx_convas, tly_convas = gt[0]+roi_size, gt[1]+roi_size
	cx = tlx_convas + int(0.5 * gt[2])
	cy = tly_convas + int(0.5 * gt[3])

	# Crop an roi_size region centered at cx, cy
	half = roi_size // 2
	roi = convas[cy-half:cy+half, cx-half:cx+half, :]

	# compute new target pos in roi window
	new_cx, new_cy = [int(i*0.5) for i in roi.shape[:-1]]
	new_x = new_cx - gt[2] // 2
	new_y = new_cx - gt[3] // 2

	return roi, [new_x, new_y, gt[2], gt[3]]

    

def extract_roi(img, GT, l_off, roi_size, r_w_scale):
	"""
	Extract Regigon of Interest 
	"""
	print(img.max(), 'origin max')
	h, w = img.shape[:2]
	win_w = GT[2]
	win_h = GT[3]
	win_lt_x = GT[0]
	win_lt_y = GT[1]

	# Center location in img
	win_cx = np.round(win_lt_x + win_w / 2 + l_off[0])
	win_cy = np.round(win_lt_y + win_h / 2 + l_off[1])

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
	print(img[y1-1:y2, x1-1:x2, :].max(), 'before bicubic resize')
	roi =  imresize(img[y1-1:y2, x1-1:x2, :], [roi_size, roi_size], interp='bicubic')
	print(roi.max(), 'after bicubic resize')
	preimg = np.zeros(img.shape[:2])
	roi_pos = [x1, y1, x2-x1+1, y2-y1+1]
	print(roi.max(), 'roi max')
	return roi, roi_pos, preimg, pad


def gen_mask_old(img_size, pos):
	"""
	Generates 2-D gaussian mask with variance proportion
	target's region

	Args:
	    img_size: 
	    pos:
	Returns:
	    masked_img:
	"""
	# sigma is consistence with the paper
	kernly, kernlx = pos[2:]
	nsig =  min(pos[2:]) / 3 #kernly / kernlx #

	# Constructs 2D gaussian 
	intervalx = (2*nsig+1.)/(kernlx)
	x = np.linspace(-nsig-intervalx/2., nsig+intervalx/2., kernlx+1)
	kern1dx = np.diff(st.norm.cdf(x))

	intervaly = (2*nsig+1.)/(kernly)
	y = np.linspace(-nsig-intervaly/2., nsig+intervaly/2., kernly+1)
	kern1dy = np.diff(st.norm.cdf(y))

	# Normalize
	kernel_raw = np.sqrt(np.outer(kern1dx, kern1dy))
	kernel = kernel_raw/kernel_raw.sum()

	print(kernel.shape, img_size)
	# Plcace into an img_size convas
	img = np.zeros(img_size)
	img[pos[1]: pos[1]+pos[3], pos[0]: pos[0]+pos[2]] = kernel
	return img


def gen_mask(im_sz, fea_sz, roi_sz, location, l_off, s):
	"""
	Generates 2D guassian masked convas with shape same as 
	fea_sz.

	Args:
		img_sz: input image size.
		fea_sz: feaure size, specifically output of sel-CNN net.
		roi_sz: roi size
		location: location parameters
		l_off: offset 
		s: scale factor
	"""
	x, y, w, h = location
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
	convas, _, _, _  = extract_roi(convas, location, l_off, roi_sz, s)
	print(convas.shape)
	convas = imresize(convas[...,0], fea_sz[:2], interp='bicubic')
	print(convas.max(), 'max convas')

	# Swap back, and normalize
	convas = convas / convas.max()
	convas = np.transpose(convas)

	return convas





def IOU_eval(groud_truth_box, predicted_box):
	"""
	Returns:
		iou: scaler
	"""

	pass

