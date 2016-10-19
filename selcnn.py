

import tensorflow as tf
import numpy as np

from utils import variable_on_cpu, variable_with_weight_decay

class SelCNN:
	def __init__(self, scope, vgg_conv_layer):
		"""
		selCNN network class. Initialize graph.

		Args: 
			name: string, name of the network.
			vgg_conv_layer: tensor, either conv4_3 or cnv5_3 layer
				of a pretrained vgg16 network
		"""
		# Initialize network
		self.scope = scope
		self.input_layer = vgg_conv_layer
		self.params = {
		'dropout_rate': 0.3,
		'k_size': [3, 3, 512, 1],
		'wd': 0.5,
		'lr_initial': 1e-9, # 1e-8 gives 438 after 200 steps, 1e-7 gives better maps?
		'lr_decay_steps': 0,
		'lr_decay_rate':  1
		}
		with tf.name_scope(scope) as scope:
			self.pre_M = self._get_pre_M()

		self.pre_M_size = self.pre_M.get_shape().as_list()[1:3]
	


	def _get_pre_M(self):
		"""Build the sel-CNN graph and returns predicted Heat map."""
		dropout_layer = tf.nn.dropout(self.input_layer, self.params['dropout_rate'])

		# Conv layer with bias 
		kernel = variable_with_weight_decay(self.scope, 'kernel',\
							self.params['k_size'], wd = self.params['wd'])
		conv = tf.nn.conv2d(dropout_layer, kernel, [1,1,1,1], 'SAME')
		bias = variable_on_cpu(self.scope,'biases', [1], tf.constant_initializer(0.1))
		pre_M = tf.nn.bias_add(conv, bias)

		# Subtract mean 
		#pre_M -= tf.reduce_mean(pre_M)
		#pre_M# /= tf.reduce_max(pre_M)
		return pre_M


	def train(self, gt_M, add_regulizer=True):
		""" Train the network on the fist frame. 

		Args:
			gt_M: tensor with shape identical to self.pre_M,
				Ground truth heatmap.
			add_regulizer: bool, True for adding L2 regulizer of the 
				kernel variables of the conv layer.

		Returns:
			train_op:
			total_losses:
			lr:
		"""
		if isinstance(gt_M, np.ndarray):
			gt_M = tf.constant(gt_M.reshape((1,gt_M.shape[0], gt_M.shape[1], 1)), dtype=tf.float32)

		gt_shape, pre_shape = gt_M.get_shape().as_list()[1:], self.pre_M.get_shape().as_list()[1:]
		assert isinstance(gt_M, tf.Tensor)
		assert gt_shape == pre_shape, \
			'Shapes are not compatiable! gt_M : {0}, pre_M : {1}'.format(
				gt_shape, pre_shape)
		
		with tf.variable_scope(self.scope) as scope:
			# Root mean square loss
			rms_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(gt_M, self.pre_M))))
			# tf.squared_difference(x, y, name=None) try this! 
			# (x-y)(x-y) 

			tf.add_to_collection('losses', rms_loss)

			# Use vanila SGD with exponentially decayed learning rate
			# Decayed_learning_rate = learning_rate *
			#                decay_rate ^ (global_step / decay_steps)
			global_step = tf.Variable(0, trainable=False)
			lr = tf.train.exponential_decay(
				self.params['lr_initial'], 
				global_step, 
				self.params['lr_decay_steps'], 
				self.params['lr_decay_rate'] , 
				name='lr')

			# Vanilia SGD with dexp decay learning rate
			optimizer = tf.train.GradientDescentOptimizer(lr)

			if add_regulizer:
				# Add L2 regularzer losses
				total_losses = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), 'total_losses')
			else:
				total_losses = rms_loss
			train_op = optimizer.minimize(total_losses, global_step=global_step)
			self.loss = total_losses
		return train_op, total_losses, lr, optimizer

	def sel_feature_maps(self, gt_M, maps, num_sel):
		""" 
		Selects saliency feature maps. 
		The change of the Loss function by the permutation
		of the feature maps dF, can be computed by a 
		two-order Taylor expansions.

		Further simplication can be done by only compute
		the diagonol part of the Hessian matrix.

		Args:
			gt_M: tensor, ground truth heat map.
			maps: tensor, conv layer of vgg.
			num_sel: int, number of selected maps.

		Returns:
			sel_maps: tensor, slected vgg conv feature maps
		"""

		pass
		