"""
Main script for FCNT tracker. 
"""
import numpy as np 
import tensorflow tf

# Import custom class and functions
from inputproducer import InputProducer
from tracker import TrackerVanilla
from vgg16 import Vgg16
from selcnn import SelCNN
from sgnet import GNet, SNet
from utils import img_with_bbox, IOU_eval


tf.app.flags.DEFINE_integer('iter_step_sel', 200,
                          """Number of steps for trainning"""
                          """selCNN networks.""")
tf.app.flags.DEFINE_integer('iter_step_sg', 50,
                          """Number of steps for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('FLAGS.num_sel', 384,
                          """Number of feature maps selected.""")
tf.app.flags.DEFINE_integer('iter_max', 200,
							"""Max iter times through imgs""")

FLAGS = tf.app.flags.FLAGS

## Define varies path
DATA_ROOT = 'data/Dog1'
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'


def train_selCNN(selCNN, gt_M, feed_dict)
	train_op, losses, lr, optimizer = l_selcnn.train(gt_M)

	# Train for iter_step_sel times
	# Inspects loss curve and pre_M visually
	for step in range(FLAGS.iter_step_sel):
		_, total_loss, pre_M, lr_ = sess.run([train_op, losses, pre_M_tensor, lr], feed_dict=feed_dict)


def train_sgNet(gnet, snet):
	"""
	Train sgnet by minimize the loss
	Loss = Lg + Ls
	where Li = |pre_Mi - gt_M|**2 + Weights_decay_term_i

	"""
	for step in range(FLAGS.iter_step_sg):
		pass
	pass


#def main(args):
## Instantiate inputProducer and retrive the first img
# with associated ground truth. 
inputProducer = InputProducer(IMG_PATH, GT_PATH)
img, gt, t  = next(inputProducer.gen_img)
roi, roi_pos, preimg, pad = inputProducer.extract_roi(img, gt)

# Predicts the first img.
sess = tf.Session()
sess.run(tf.initialize_all_variables())
vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
vgg.print_prob(roi, sess)

## At t=0. Perform the following:
# 1. Train selCNN network for both local and gloabl feature maps
# 2. Train G and S networks.
assert t == 0:

lselCNN = SelCNN('sel_local', vgg.conv4_3)
gselCNN = SelCNN('sel_global', vgg.con5_3)

lgt_M = inputProducer.gen_mask(lselCNN.pre_M_size)
ggt_M = inputProducer.gen_mask(gselCNN.pre_M_size)

# Train selCNN networks with first frame roi
feed_dict = {vgg.imgs: roi}
train_selCNN(lselCNN, lgt_M, feed_dict)
train_selCNN(gselCNN, ggt_M, feed_dict)

# Perform saliency maps selection 
s_sel_maps = lselCNN.sel_feature_maps(lgt_M, vgg.conv4_3, FLAGS.num_sel)
g_sel_maps = gselCNN.sel_feature_maps(ggt_M, vgg.conv5_3, FLAGS.num_sel)

# Instantiate G and S networks by sending selected saliency maps.
gnet = GNet('GNet', g_sel_maps)
snet = SNet('SNet', s_sel_maps)

# Train G and S nets by minimizing a composite loss.
train_sgNet(gnet, snet)

## At t>0. Perform target localization and distracter detection at every frame,
## perform SNget adaptive update every 20 frames, perform SNet discrimtive 
## update if distracter detection return True.

# Instantiate Tracker object and initialize it with lgt_M.
tracker = TrackerVanilla(lgt_M, gt)

# Iter imgs
gt_last = gt 
for i in range(FLAGS.iter_max):
	# Gnerates next frame infos
	img, gt_cur, t  = next(inputProducer.gen_img)

	## Crop a rectangle ROI region centered at last target location.
	roi, roi_pos, preimg, pad = inputProducer.extract_roi(img, gt_last)
	
	## Perform Target localiation predicted by GNet
	# Get heat map predicted by GNet
	feed_dict = {vgg.imgs : roi}
	pre_M = sess.run(gnet.pre_M, feed_dict=feed_dict)

	if i % 20 == 0:
		# Use predicted heat map to adaptive finetune SNet.
		snet.adaptive_finetune(sess, pre_M)

	# Localize target use monte carlo method.
	tracker.draw_particles(gt_last)
	pre_loc = tracker.predict_location(pre_M)

	# Performs distracter detecion operation,
	if tracker.distracted():
		# if detects distracters, then update 
		# SNet using descrimtive method.
		snet.descrimtive_finetune(sess, pre_M)
		pre_M = sess.run(snet.pre_M, feed_dict=feed_dict)

		# Use location predicted by SNet.
		pre_loc = tracker.predict_location(pre_M)
	
	# Set predicted location to be the next frame's ground truth
	gt_last = pre_loc

	# Draw bbox on image. And print associated IoU score.
	img_with_bbox(img, pre_loc, gt_cur)
	IOU_eval()

