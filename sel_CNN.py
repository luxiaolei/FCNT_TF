

class selCNN:
	def __init__(self):
		"""
		selCNN network class. 
		"""
		# Initialize network

	def _variable_on_cpu(name, shape, initializer):
		"""Helper to create a Variable stored on CPU memory.

		Args:
		  name: name of the variable
		  shape: list of ints
		  initializer: initializer for Variable

		Returns:
		  Variable Tensor
		"""
		dtype = tf.float32
		with tf.device('/cpu:0'):
			variable = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
		return variable

	def variable_with_weight_decay(name, shape, stddev=1e-3, wd=None):
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
		variable = _variable_on_cpu(
								name, 
								shape,
								initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
		if wd is not None:
			weight_decay = tf.mul(tf.nn.l2_loss(variable), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return variable

	def train(self):
		# Train for the fist frame
		pass

	def gen_feature_maps(self):
		# Evaluation method of the network
		pass
		