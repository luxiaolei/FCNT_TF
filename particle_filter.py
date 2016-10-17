







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
	def __init__(self, init_location):
		
		self.location = init_location
		self.params = self._init_params(init_location)
		self.params = {
		'aff_sig': [10,10,.004,.00,0.00,0], # variances of affine parameters
		'p_sz': 64, # particle size?
		'p_num': 700,  # num of particles
		'mv_thr': 0.1, 
		'up_thr': 0.35,
		'roi_scale': 2}

	def _init_params(self, init_location):
		"""Initialize tracker's parameters"""

		params = {'p_sz': 64, 'p_num': 700, 'min_conf': 0.5, 
				'mv_thr': 0.1, 'up_thr': 0.35, 'roi_scale': 2}
		diag_s = np.ceil((init_location[2]**2 + init_location[3]**2)**0.5/7)
		params['aff_sig'] = [diag_s, diag_s, 0.004, 0.0, 0.0, 0]
		params['ratio'] = init_location[2] / params['p_sz']
		
		return params

	def draw_particles(self, aff_params):
		"""
		Generates particles according to 
		P(Xt|Xt−1) = N (Xt; Xt−1, Ψ)

		Args:
			aff_params: [cx, cy, w/p_sz, 0, h/w, 0],
				affine parameters, see class doc string for 
				specific element definition.

		Returns:
			aff_params_M : 6 * self.p_num size matrix,
				where rows are updated randomly drawed affine 
				params, columns repersent each particles. 
		"""
		# Construct an 6*p_num size matrix with with each 
		# column repersents one particle
		aff_params_M = np.kron(np.ones((self.params['p_num'],1)), np.array(aff_params)).T

		# First onstruct a 6*p_num size normal distribution with 
		# mean 0 and sigma 1
		rand_norml_M = np.array([np.random.standard_normal(6) for _ in range(self.params['p_num'])]).T

		# Then construct a affine sigma matrix
		aff_sig_M = np.kron(np.ones((p_num, 1)), self.params['aff_sig']).T

		# Update particles 
		aff_params_M += rand_norml_M * aff_sig_M

		return aff_params_M

	def compute_conf(self):
		pass

	def warp_img(self, img):
		pass

	def distracted(self):

		return False

	def update_location(self, new_location):
		self.location = new_location



def loc_2_geoparam():
	"""
	Convert location to geoparameters
	"""
	pass

def geoparam_2_loc():
	"""
	Convert geoparameters to location
	"""
	pass


