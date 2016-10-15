

import tensorflow as tf
import numpy as np

from utils import variable_on_cpu, variable_with_weight_decay

class selCNN:
	def __init__(self, name, vgg_conv_layer):
		"""
		selCNN network class. 
		"""
		# Initialize network
		self.name = name
		self.input_layer = vgg_conv_layer
		self.params = {
		'dropout_rate': 0.3,
		'k_size': [3, 3, 512, 1],
		'wd': 0.1,
		'lr': 1e-9
		}
		self.pre_M = self._get_pre_M()
	


	def _get_pre_M(self):
		dropout_layer = tf.nn.dropout(self.input_layer, self.params['dropout_rate'])

		# Conv layer with bias 
		kernel = variable_with_weight_decay(self.name+'/kernel',\
							self.params['k_size'], wd = self.params['wd'])
		conv = tf.nn.conv2d(dropout_layer, kernel, [1,1,1,1], 'SAME')
		bias = variable_on_cpu(self.name+'/biases', [1], tf.constant_initializer(0.1))
		pre_M = tf.nn.bias_add(conv, bias)

		# Subtract mean 
		pre_M -= tf.reduce_mean(pre_M)
		return pre_M


	def train(self, gt_M):
		# Train for the fist frame
		if isinstance(gt_M, np.ndarray):
			gt_M = tf.constant(gt_M.reshape((1,gt_M.shape[0], gt_M.shape[1], 1)), dtype=tf.float32)

		assert isinstance(gt_M, tf.Tensor)
		assert gt_M.get_shape() == self.pre_M.get_shape(), 'Shapes are not compatiable!'
		
		# Root mean square loss
		rms_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(gt_M, self.pre_M))))
		tf.add_to_collection('losses', rms_loss)

		# Add L2 regularzer losses
		total_losses = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), 'total_losses')
		optimizer = tf.train.GradientDescentOptimizer(self.params['lr'])
		train_op = optimizer.minimize(total_losses)
		return train_op, total_losses

	def gen_feature_maps(self):
		# Evaluation method of the network
		pass
		