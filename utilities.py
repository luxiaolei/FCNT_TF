



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


