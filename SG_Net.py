


class SGNet:
	def __init__(self):
		"""
		Base calss for SGNet, defines the network structure
		"""

		
	def _build(self, name):
		"""
		Define Structure
		"""
		pass


class GNet(SGNet):
	def __init__(self):
		"""
		Fixed params once trained in the first frame
		"""
		super(SGNet.__init__)

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

		
