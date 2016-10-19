
import tensorflow as tf
import numpy as np 

from utils import variable_on_cpu, variable_with_weight_decay


class SGNet:

	# Define class level optimizer
	lr = 1e-6
	optimizer = tf.train.GradientDescentOptimizer(lr)

    def __init__(self, scope, vgg_conv_shape):
        """
        Base calss for SGNet, defines the network structure
        """
        self.scope = scope
        self.params = {
        'num_fms': 200, # number of selected featrue maps, inputs of the network
        'wd': 0.5, # L2 regulization coefficient
        }
        self.variables = []
        with tf.variable_scope(scope) as scope:
            self.pre_M = self._build_graph(vgg_conv_shape)

    def _build_graph(self, vgg_conv_shape):
        """
        Define Structure. 
        The first additional convolutional
        layer has convolutional kernels of size 9×9 and outputs
        36 feature maps as the input to the next layer. The second
        additional convolutional layer has kernels of size 5 × 5
        and outputs the foreground heat map of the input image.
        ReLU is chosen as the nonlinearity for these two layers.

        Args:
            vgg_conv_shape: 
        Returns:
            conv2: 
        """
        self.variables = []
        self.kernel_weights = []
        out_num = vgg_conv_shape[-1]
        self.input_maps = tf.placeholder(tf.float32, shape=vgg_conv_shape,
            name='selected_maps')
        #assert vgg_conv_shape[-1] == self.params['num_fms']
        
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([9,9,out_num,36], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.input_maps, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[36], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(out, name=scope)
            self.variables += [kernel, biases]
            self.kernel_weights += [kernel]
            print(conv1.get_shape().as_list(), 'conv1 shape')


        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5,5,36,1], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv1, kernel , [1, 1, 1, 1], padding='SAME')
            print(conv.get_shape().as_list(), 'conv shape')
            biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(out, name=scope)
            self.variables += [kernel, biases]
            self.kernel_weights += [kernel]

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
            beta = tf.constant(self.params['wd'], name='beta')
            loss_rms = tf.reduce_mean(tf.square(tf.sub(gt_M, self.pre_M))) 
            loss_wd = [tf.reduce_mean(tf.square(w)) for w in self.kernel_weights]
            loss_wd = beta * tf.add_n(loss_wd)
            total_loss = loss_rms + loss_wd
        return total_loss

	@classmethod
	def eadge_RP():
		"""
		This method propose a series of ROI along eadges
		of a given frame. This should be called when particle 
		confidence below a critical value, which possibly accounts
		for object re-appearance.
		"""
		pass



class GNet(SGNet):
	def __init__(self, scope, vgg_conv_shape):
		"""
		Fixed params once trained in the first frame
		"""
		super(GNet, self).__init__(scope, vgg_conv_shape)






class SNet(SGNet, scope, vgg_conv_shape):
	def __init__(self):
		"""
		Initialized in the first frame
		"""
		super(SNet, self).__init__(scope, vgg_conv_shape)

	def adaptive_finetune(self, sess, updated_gt_M):
		"""Finetune SNet with updated_gt_M."""

		pass

	def descrimtive_finetune(self, sess, init_gt_M, cur_):
		pass

		
