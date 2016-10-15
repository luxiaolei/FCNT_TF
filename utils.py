
import numpy as np
import skimage


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
	m,n = [(ss-1.)/2. for ss in shape]
	y,x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	h = h / h.max()
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

    

def extract_roi(image,location, scale):
	"""
	Extract Regigon of Interest 
	"""
	pass

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


