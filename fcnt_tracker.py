"""
Main script for FCNT tracker. 
"""
#%%
# Import custom class and functions
from inputproducer import InputProducer
from tracker import TrackerVanilla
from vgg16 import Vgg16
from selcnn import SelCNN
from sgnet import GNet, SNet
from utils import img_with_bbox, IOU_eval

import numpy as np 
import tensorflow as tf

import os

#%%
tf.app.flags.DEFINE_integer('iter_step_sel', 200,
                          """Number of steps for trainning"""
                          """selCNN networks.""")
tf.app.flags.DEFINE_integer('iter_step_sg', 50,
                          """Number of steps for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('num_sel', 384,
                          """Number of feature maps selected.""")
tf.app.flags.DEFINE_integer('iter_max', 200,
							"""Max iter times through imgs""")

FLAGS = tf.app.flags.FLAGS

## Define varies path
DATA_ROOT = 'data/Dog1'
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'


def train_selCNN(sess, selCNN, gt_M_sz, feed_dict):
	# Initialize variables
	global_step = tf.Variable(0, trainable=False)
	selCNN_vars = selCNN.variables 
	init_vars_op = tf.initialize_variables(selCNN_vars + [global_step], name='init_selCNN')
	sess.run(init_vars_op)

	# Retrive trainning op
	train_op, losses, lr, optimizer = selCNN.train_op(gt_M_sz, global_step)
	print(sess.run(tf.report_uninitialized_variables()))
	# Train for iter_step_sel times
	# Inspects loss curve and pre_M visually
	for step in range(FLAGS.iter_step_sel):
		_, total_loss, lr_ = sess.run([train_op, losses, lr], feed_dict=feed_dict)
		print(total_loss)


def train_sgNet(sess, gnet, snet, sgt_M, ggt_M, feed_dict):
	"""
	Train sgnet by minimize the loss
	Loss = Lg + Ls
	where Li = |pre_Mi - gt_M|**2 + Weights_decay_term_i

	"""
	# Initialize sgNet variables
	sgNet_vars = gnet.variables + snet.variables
	init_SGNet_vars_op = tf.initialize_variables(sgNet_vars, name='init_sgNet')
	sess.run(init_SGNet_vars_op)

	# Define composite loss
	total_losses = snet.loss(sgt_M) + gnet.loss(ggt_M)

	# Define trainning op
	optimizer = tf.train.GradientDescentOptimizer(1e-6)
	train_op = optimizer.minimize(total_losses, var_list= sgNet_vars)

	for step in range(FLAGS.iter_step_sg):
		loss, _ = sess.run([total_losses, train_op], feed_dict = feed_dict)
		print(loss)



def gen_mask_phi(img_sz, loc):
	x,y,w,h = loc
	phi = np.zeros(img_sz)
	phi[[y-int(0.5*h): y+int(0.5*h), x-int(0.5*w):x+int(0.5*w)]] = 1
	return phi


def main(args):
	## Instantiate inputProducer and retrive the first img
	# with associated ground truth. 
	inputProducer = InputProducer(IMG_PATH, GT_PATH)
	img, gt, t  = next(inputProducer.gen_img)
	roi_t0, _, _ = inputProducer.extract_roi(img, gt)

	# Predicts the first img.
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
	vgg.print_prob(roi_t0, sess)

	## At t=0. Perform the following:
	# 1. Train selCNN network for both local and gloabl feature maps
	# 2. Train G and S networks.
	assert t == 0:

	lselCNN = SelCNN('sel_local', vgg.conv4_3)
	gselCNN = SelCNN('sel_global', vgg.con5_3)

	# Gen anotated mask for target arear
	sgt_M = inputProducer.gen_mask(lselCNN.pre_M_size)
	ggt_M = inputProducer.gen_mask(gselCNN.pre_M_size)

	## Train selCNN networks with first frame roi
	# reshape gt_M for compatabilities
	sgt_M = sgt_M[np.newaxis,:,:,np.newaxis]
	ggt_M = ggt_M[np.newaxis,:,:,np.newaxis]

	feed_dict = {vgg.imgs: [roi_t0], 
				lselCNN.gt_M: sgt_M,
				gselCNN.gt_M: ggt_M} # corrpus the other nets?

	train_selCNN(sess, lselCNN, sgt_M.shape, feed_dict)
	train_selCNN(sess, gselCNN, ggt_M.shape, feed_dict)

	# Perform saliency maps selection 
	s_sel_maps, s_idx = lselCNN.sel_feature_maps(sess, vgg.conv4_3, feed_dict,FLAGS.num_sel)
	g_sel_maps, g_idx = gselCNN.sel_feature_maps(sess, vgg.conv5_3, feed_dict,FLAGS.num_sel)

	assert isinstance(s_sel_maps, np.ndarray)
	assert isinstance(g_sel_maps, np.ndarray)
	assert len(s_sel_maps.shape) == 4

	# Instantiate G and S networks.
	gnet = GNet('GNet', s_sel_maps.shape)
	snet = SNet('SNet', s_sel_maps.shape)

	## Train G and S nets by minimizing a composite loss.
	## with feeding selected saliency maps for each networks.
	feed_dict = {gnet.input_maps: g_sel_maps, snet.input_maps: s_sel_maps}
	train_sgNet(sess, gnet, snet, sgt_M, ggt_M, feed_dict)
	s_sel_maps_t0 = s_sel_maps


	## At t>0. Perform target localization and distracter detection at every frame,
	## perform SNget adaptive update every 20 frames, perform SNet discrimtive 
	## update if distracter detection return True.

	# Instantiate Tracker object and initialize it with sgt_M.
	tracker = TrackerVanilla(sgt_M, gt)

	# Iter imgs
	gt_last = gt 
	for i in range(FLAGS.iter_max):
		# Gnerates next frame infos
		img, gt_cur, t  = next(inputProducer.gen_img)

		## Crop a rectangle ROI region centered at last target location.
		roi, _, resize_factor = inputProducer.extract_roi(img, gt_last)
		
		## Perform Target localiation predicted by GNet
		# Get heat map predicted by GNet
		feed_dict_vgg = {vgg.imgs : [roi]}
		s_maps, g_maps = sess.run([vgg.conv4_3, vgg.conv5_3], feed_dict=feed_dict_vgg)
		s_sel_maps = s_maps[s_idx] # np.ndarray, shape = [1,28,28,num_sel]?
		g_sel_maps = g_maps[g_idx]

		feed_dict_g = { gnet.input_maps: g_sel_maps}
		pre_M = sess.run(gnet.pre_M, feed_dict=feed_dict_g)
		tracker.pre_M_q.push(pre_M)

		if i % 20 == 0:
			# Retrive the most confident result within the intervening frames
			best_M = tracker.gen_best_M()

			# Use the best predicted heat map to adaptive finetune SNet.
			snet.adaptive_finetune(sess, best_M)

		# Localize target with monte carlo sampling.
		tracker.draw_particles()
		pre_loc = tracker.predict_location(pre_M, gt_last, resize_factor, t)

		# Performs distracter detecion.
		if tracker.distracted():
			# if detects distracters, then update 
			# SNet using descrimtive loss.
			# gen mask
			phi = gen_mask_phi(roi.shape, pre_loc)
			snet.descrimtive_finetune(sess, s_sel_maps_t0, sgt_M, roi, s_sel_maps, phi)
			pre_M = sess.run(snet.pre_M, feed_dict=feed_dict)

			# Use location predicted by SNet.
			pre_loc = tracker.predict_location(pre_M)
		
		# Set predicted location to be the next frame's ground truth
		gt_last = pre_loc

		# Draw bbox on image. And print associated IoU score.
		img_with_bbox(img, pre_loc, gt_cur)
		IOU_eval()

if __name__=='__main__':
	tf.app.run()

