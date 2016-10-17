
import tensorflow as tf
import numpy as np 

from utils import variable_on_cpu, variable_with_weight_decay


class SGNet:

	# Define class level optimizer

	def __init__(self, scope, feature_maps):
		"""
		Base calss for SGNet, defines the network structure
		"""
		self.scope = scope
		self.params = {
		'num_fms': 200, # number of selected featrue maps, inputs of the network
		'wd': 0.5, # L2 regulization coefficient
		}
		with tf.name_scope(scope) as scope:
			self.pre_M = self._build_graph(feature_maps)
		
	def _build_graph(self, feature_maps):
		"""
		Define Structure. 
		The first additional convolutional
		layer has convolutional kernels of size 9×9 and outputs
		36 feature maps as the input to the next layer. The second
		additional convolutional layer has kernels of size 5 × 5
		and outputs the foreground heat map of the input image.
		ReLU is chosen as the nonlinearity for these two layers.

		Args:
			feature_maps: 
		Returns:
			conv2: 
		"""
		assert isinstant(feature_maps, tf.Tensor)
		assert feature_maps.get_shape().as_list()[-1] == self.params['num_fms']

		with tf.name_scope('Conv1') as scope:
			kernel = variable_with_weight_decay('kernel', 
				[9,9,self.params['num_fms'],36], wd = self.params['wd'])
			conv = tf.nn.conv2d(feature_maps, kernel, [1,1,1,1], 'SAME')
			bias = variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
			out = tf.nn.bias_add(conv, bias)
			conv1 = tf.nn.relu(out, name=scope)

		with tf.name_scope('Conv2') as scope:
			kernel = variable_with_weight_decay('kernel', 
				[5,5,36,1], wd = self.params['wd'])
			conv = tf.nn.conv2d(conv1, kernel, [1,1,1,1], 'SAME')
			bias = variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
			out = tf.nn.bias_add(conv, bias)
			conv2 = tf.nn.relu(out, name=scope)
		print('Shape of the out put heat map for %s is %s'%(self.scope, conv2.get_shape().as_list()))
		return conv2

	def loss(self, gt_M):
		"""Returns Losses for the current network.

		Args:
			gt_M: Tensor, ground truth heat map.

		Returns:
			Loss: 
		"""

		# Assertion
		with tf.name_scope(self.scope) as scope:

			#





class GNet(SGNet):
	def __init__(self, scope, feature_maps):
		"""
		Fixed params once trained in the first frame
		"""
		super(GNet, self).__init__(scope, feature_maps)

	def distracter_detection(self, gt):

		return False





class SNet(SGNet):
	def __init__(self):
		"""
		Initialized in the first frame
		"""

	def adaptive_finetune(self):
		pass

	def descrimtive_finetune(self):
		pass

		
