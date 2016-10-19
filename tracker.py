

from queue import Queue
from scipy.misc import imresize
from operator import add

def compute_conf(roi, loc_p):
	"""Helper func for computing confidence"""
	cx,cy,w,h = loc_p
	conf = np.sum(roi[y-int(0.5*h): y+int(0.5*h), x-int(0.5*w):x+int(0.5*w)])
	return conf


class Tracker:
	"""
	Generic tracking model. A location is represented by an affine transformation (e.g., Xt−1), which warps the
	coordinate system so that the target lies within the unit square. Particles representing possible target locations Xt, 
	at time t are sampled according to P(Xt|Xt−1), which in this case is a diagonal-covariance Gaussian centered at Xt−1.
	
	Where:
	Xt = (xt, yt, θt, st, αt, φt)
	denote x, y translation, rotation angle, scale, aspect ratio, and skew direction at time t.

	P(Xt|Xt−1) = N (Xt; Xt−1, Ψ)
	where Ψ is a diagonal covariance matrix whose elements are the corresponding variances of affine parameters, assumes the variance of each affine parameter does not change over time

	See 3.3.1 Dynamic model in http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf for reference

	Particle filter calss"""
	def __init__(self, init_gt_M, init_location,):
		
		self.init_gt_M = init_gt_M
		self.conf_records = Queue(maxsize=20)
		self.last_two_location = Queue(maxsize=2)

		self.location = init_location
		self.params = self._init_params(init_location)



	def _init_params(self, init_location):
		"""Initialize tracker's parameters"""

		params = {'p_sz': 64, 'p_num': 700, 'min_conf': 0.2, 
				'mv_thr': 0.1, 'up_thr': 0.35, 'roi_scale': 2}
		diag_s = np.ceil((init_location[2]**2 + init_location[3]**2)**0.5/7)
		params['aff_sig'] = [diag_s, diag_s, 0.004, 0.0, 0.0, 0]
		params['ratio'] = init_location[2] / params['p_sz']
		
		return params


	def draw_particles(self):
		"""
		Generates particles according to 
		P(Xt|Xt−1) = N (Xt; Xt−1, Ψ)

		Args:
			aff_params: affine parameters, see class doc string for 
				specific element definition.
				[cx, cy, w/p_sz, 0, h/w, 0] for 6 degrees of freendom
				[tlx, tly, w, h] for 4 degrees of freedom.
				.
		Returns:
			aff_params_M : self.p_num*dof size matrix,
				where rows are updated randomly drawed affine 
				params, columns repersent each particles. 
		"""
		# Define degrees of freedom 
		dof = len(self.params['aff_sig'])

		# Construct an p_num*6 size matrix with with each 
		# column repersents one particle

		#aff_params_M = np.kron(np.ones((self.params['p_num'],1)), np.array(aff_params))

		# First onstruct a p_num*dof size normal distribution with 
		# mean 0 and sigma 1
		rand_norml_M = np.array([np.random.standard_normal(dof) for _ in range(self.params['p_num'])])

		# Then construct a affine sigma matrix
		aff_sig_M = np.kron(np.ones((self.params['p_num'], 1)), self.params['aff_sig'])

		# Update particles 
		self.aff_params_M = rand_norml_M * aff_sig_M



	def predict_location(self, pre_M, gt_last, resize_factor, t):
		"""
		Predict location for each particle. It is calculated by
		1. compute the confidence of the i-th candidate, which is 
			the summation of all the heatmap values within the candidate region.
		2. the candidate with the highest confidence value is predicted as target.

		Args:
			img_siz: tuple(image height, image width)
			pre_M: predicted heat map
			t: index of current frame
		"""
		# transform self.aff_params_M to location_M with each column 
		# repersent [cx, cy, w, h] in the pre_M heat map
		loc_M = np.zeros(self.aff_params_M.shape)
		tlx, tly, w, h = gt
		cx, cy = roi_size // 2, roi_size // 2
		loc_M[:, 0] = cx
		loc_M[:, 1] = cy
		loc_M[:, 2] = resize_factor * w 
		loc_M[:, 3] = resize_factor * h
		loc_M += self.aff_params_M
		loc_M = loc_M.astype(np.int)

		# Upsampling pre_M bicubicly to roi_size
		pre_M_resized = imresize(pre_M, [roi_size, roi_size], interp='bicubic')

		# Compute conf for each particle 
		conf_lsit = []
		for p_i_loc in loc_M:
			conf_i = self.compute_conf(pre_M_resized, p_i_loc)
			conf_lsit += [conf_i]

		# Get index and conf score of of the most confident one
		idx = np.argmax(conf_lsit)
		self.cur_best_conf = conf_lsit[idx]

		# Get the corresponding aff_param which is then
		# used to predicted the cureent best location
		best_aff =  self.aff_params_M[idx]
		self.pre_location = self.aff2loc(gt_last, best_aff)

		# Stack into records queue


		return self.pre_location


	def get_most_conf_M(self):
		"""Returns the most confidence heat maps."""

		# Pull self.conf_records all out, and retrive 
		# the most confident heat map. 

		return updated_gt_M


	def linear_prediction(self):
		"""
		Predicts current location linnearly according
		to las two frames location. This may boost the 
		robustnesss of obejct occlusion.
		"""
		pass

	def distracted(self):
		"""Distracter detection."""

		# up-sampling pre_M

		# Compute confidence according to 
		# S = with_in / with_out
		if self.cur_best_conf <= self.params['min_conf']:
			return True
		else:
			return False
			
	@classmethod
	def compute_conf(roi, loc_p):
		"""Helper func for computing confidence"""
	    cx,cy,w,h = loc_p
	    conf = np.sum(roi[y-int(0.5*h): y+int(0.5*h), x-int(0.5*w):x+int(0.5*w)])
	    return conf

	@classmethod
	def affgeo2loc(las_loc, aff_param):
		"""Convert affine params to location."""
		assert len(aff_param)==4, 'This method only works for dof 4 aff space.'
		cur_loc = map(add, las_loc, aff_param)
		return cur_loc


class TrackerVanilla(Tracker):
	"""Vanilla tracker

		The covariance matrix has only 4 degrees of freedom,
		specified by vertical, horizontal translation of the central
		point, variance of the width, variance of the w/h ratio.

		The corrresponding actual senarios are object replacment,
		object zoom in/out, object rotaion. Should be sufficient 
		for most cases of car tracking.

	"""
	def __init__(self, init_location):
		super(TrackerVanilla, self).__init__(init_location)
		self._update_params()

	def _update_params(self):
		"""Update aff_sig param."""
		self.params['aff_sig'] = []


	def draw_particles(self, last_location):
		"""
		The covariance matrix has only 4 degrees of freedom,
		specified by vertical, horizontal translation of the central
		point, variance of the width, variance of the w/h ratio.

		The corrresponding actual senarios are object replacment,
		object zoom in/out, object rotaion. Should be sufficient 
		for most cases of car tracking.

		Args: 
			last_location: [tlx, tly, w, h]
		Returns:
			particle_M: 
		"""

		return particle_M










		

