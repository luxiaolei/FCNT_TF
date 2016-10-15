"""
Load pretrained vgg16 model
"""

import tensorflow as tf

TFMODEL_PATH = 'vgg16.tfmodel'

class Vgg16(object):
	"""docstring for Vgg16"""
	def __init__(self):
		"""
		
		"""
		self.inputs = tf.placeholder("float", [1, 224, 224, 3], name='inputs')
		self.graph = self._get_graph()
		self.softmax = self.graph.get_tensor_by_name("vgg16/prob:0")
		self.conv4_3 = self.graph.get_tensor_by_name('vgg16/conv4_3/Relu:0')
		self.conv5_3 = self.graph.get_tensor_by_name('vgg16/conv5_3/Relu:0')
		self.nodes = [n.name for n in self.graph.as_graph_def().node]
	
	def _get_graph(self):
		# Get graph from saved model
		with open(TFMODEL_PATH, mode='rb') as f:
			fileContent = f.read()

		graph_def = tf.GraphDef()
		graph_def.ParseFromString(fileContent)

		# Change input tensor
		tf.import_graph_def(graph_def, input_map={ "images": self.inputs }, name='vgg16')

		graph = tf.get_default_graph()
		return graph

		
