from __future__ import absolute_import

from itertools import chain
import math

from six import iteritems
from six.moves import map, range, zip
from six import iterkeys

import copy
try:
	from numbers import Number
except ImportError:
	Number = (int, long, float, complex)

inf = float('inf')

class Gaussian(object):
	#: Precision, the inverse of the variance.
	pi = 0
	#: Precision adjusted mean, the precision multiplied by the mean.
	tau = 0

	def __init__(self, mu=None, sigma=None, pi=0, tau=0):
		if mu is not None:
			if sigma is None:
				raise TypeError('sigma argument is needed')
			elif sigma == 0:
				raise ValueError('sigma**2 should be greater than 0')
			pi = sigma ** -2
			tau = pi * mu
		self.pi = pi
		self.tau = tau

	@property
	def mu(self):
		return self.pi and self.tau / self.pi

	@property
	def sigma(self):
		return math.sqrt(1 / self.pi) if self.pi else inf

	def __mul__(self, other):
		pi, tau = self.pi + other.pi, self.tau + other.tau
		return Gaussian(pi=pi, tau=tau)

	def __truediv__(self, other):
		pi, tau = self.pi - other.pi, self.tau - other.tau
		return Gaussian(pi=pi, tau=tau)

	__div__ = __truediv__  # for Python 2

	def __eq__(self, other):
		return self.pi == other.pi and self.tau == other.tau

	def __lt__(self, other):
		return self.mu < other.mu

	def __le__(self, other):
		return self.mu <= other.mu

	def __gt__(self, other):
		return self.mu > other.mu

	def __ge__(self, other):
		return self.mu >= other.mu

	def __repr__(self):
		return 'N(mu={:.3f}, sigma={:.3f})'.format(self.mu, self.sigma)

	def _repr_latex_(self):
		latex = r'\mathcal{{ N }}( {:.3f}, {:.3f}^2 )'.format(self.mu, self.sigma)
		return '$%s$' % latex

class Matrix(list):
	def __init__(self, src, height=None, width=None):
		if callable(src):
			f, src = src, {}
			size = [height, width]
			if not height:
				def set_height(height):
					size[0] = height
				size[0] = set_height
			if not width:
				def set_width(width):
					size[1] = width
				size[1] = set_width
			try:
				for (r, c), val in f(*size):
					src[r, c] = val
			except TypeError:
				raise TypeError('A callable src must return an interable '
								'which generates a tuple containing '
								'coordinate and value')
			height, width = tuple(size)
			if height is None or width is None:
				raise TypeError('A callable src must call set_height and '
								'set_width if the size is non-deterministic')
		if isinstance(src, list):
			is_number = lambda x: isinstance(x, Number)
			unique_col_sizes = set(map(len, src))
			everything_are_number = filter(is_number, sum(src, []))
			if len(unique_col_sizes) != 1 or not everything_are_number:
				raise ValueError('src must be a rectangular array of numbers')
			two_dimensional_array = src
		elif isinstance(src, dict):
			if not height or not width:
				w = h = 0
				for r, c in iterkeys(src):
					if not height:
						h = max(h, r + 1)
					if not width:
						w = max(w, c + 1)
				if not height:
					height = h
				if not width:
					width = w
			two_dimensional_array = []
			for r in range(height):
				row = []
				two_dimensional_array.append(row)
				for c in range(width):
					row.append(src.get((r, c), 0))
		else:
			raise TypeError('src must be a list or dict or callable')
		super(Matrix, self).__init__(two_dimensional_array)

	@property
	def height(self):
		return len(self)

	@property
	def width(self):
		return len(self[0])

	def transpose(self):
		height, width = self.height, self.width
		src = {}
		for c in range(width):
			for r in range(height):
				src[c, r] = self[r][c]
		return type(self)(src, height=width, width=height)

	def minor(self, row_n, col_n):
		height, width = self.height, self.width
		if not (0 <= row_n < height):
			raise ValueError('row_n should be between 0 and %d' % height)
		elif not (0 <= col_n < width):
			raise ValueError('col_n should be between 0 and %d' % width)
		two_dimensional_array = []
		for r in range(height):
			if r == row_n:
				continue
			row = []
			two_dimensional_array.append(row)
			for c in range(width):
				if c == col_n:
					continue
				row.append(self[r][c])
		return type(self)(two_dimensional_array)

	def determinant(self):
		height, width = self.height, self.width
		if height != width:
			raise ValueError('Only square matrix can calculate a determinant')
		tmp, rv = copy.deepcopy(self), 1.
		for c in range(width - 1, 0, -1):
			pivot, r = max((abs(tmp[r][c]), r) for r in range(c + 1))
			pivot = tmp[r][c]
			if not pivot:
				return 0.
			tmp[r], tmp[c] = tmp[c], tmp[r]
			if r != c:
				rv = -rv
			rv *= pivot
			fact = -1. / pivot
			for r in range(c):
				f = fact * tmp[r][c]
				for x in range(c):
					tmp[r][x] += f * tmp[c][x]
		return rv * tmp[0][0]

	def adjugate(self):
		height, width = self.height, self.width
		if height != width:
			raise ValueError('Only square matrix can be adjugated')
		if height == 2:
			a, b = self[0][0], self[0][1]
			c, d = self[1][0], self[1][1]
			return type(self)([[d, -b], [-c, a]])
		src = {}
		for r in range(height):
			for c in range(width):
				sign = -1 if (r + c) % 2 else 1
				src[r, c] = self.minor(r, c).determinant() * sign
		return type(self)(src, height, width)

	def inverse(self):
		if self.height == self.width == 1:
			return type(self)([[1. / self[0][0]]])
		return (1. / self.determinant()) * self.adjugate()

	def __add__(self, other):
		height, width = self.height, self.width
		if (height, width) != (other.height, other.width):
			raise ValueError('Must be same size')
		src = {}
		for r in range(height):
			for c in range(width):
				src[r, c] = self[r][c] + other[r][c]
		return type(self)(src, height, width)

	def __mul__(self, other):
		if self.width != other.height:
			raise ValueError('Bad size')
		height, width = self.height, other.width
		src = {}
		for r in range(height):
			for c in range(width):
				src[r, c] = sum(self[r][x] * other[x][c]
								for x in range(self.width))
		return type(self)(src, height, width)

	def __rmul__(self, other):
		if not isinstance(other, Number):
			raise TypeError('The operand should be a number')
		height, width = self.height, self.width
		src = {}
		for r in range(height):
			for c in range(width):
				src[r, c] = other * self[r][c]
		return type(self)(src, height, width)

	def __repr__(self):
		return '{}({})'.format(type(self).__name__, super(Matrix, self).__repr__())

	def _repr_latex_(self):
		rows = [' && '.join(['%.3f' % cell for cell in row]) for row in self]
		latex = r'\begin{matrix} %s \end{matrix}' % r'\\'.join(rows)
		return '$%s$' % latex

def _gen_erfcinv(erfc, math=math):
	def erfcinv(y):
		"""The inverse function of erfc."""
		if y >= 2:
			return -100.
		elif y <= 0:
			return 100.
		zero_point = y < 1
		if not zero_point:
			y = 2 - y
		t = math.sqrt(-2 * math.log(y / 2.))
		x = -0.70711 * \
			((2.30753 + t * 0.27061) / (1. + t * (0.99229 + t * 0.04481)) - t)
		for i in range(2):
			err = erfc(x) - y
			x += err / (1.12837916709551257 * math.exp(-(x ** 2)) - x * err)
		return x if zero_point else -x
	return erfcinv

def _gen_ppf(erfc, math=math):
	erfcinv = _gen_erfcinv(erfc, math)
	def ppf(x, mu=0, sigma=1):
		return mu - sigma * math.sqrt(2) * erfcinv(2 * x)
	return ppf

def erfc(x):
	z = abs(x)
	t = 1. / (1. + z / 2.)
	r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
		0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
			0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
				-0.82215223 + t * 0.17087277
			)))
		)))
	)))
	return 2. - r if x < 0 else r

def cdf(x, mu=0, sigma=1):
	return 0.5 * erfc(-(x - mu) / (sigma * math.sqrt(2)))


def pdf(x, mu=0, sigma=1):
	return (1 / math.sqrt(2 * math.pi) * abs(sigma) *
			math.exp(-(((x - mu) / abs(sigma)) ** 2 / 2)))

ppf = _gen_ppf(erfc)

def choose_backend(backend):
	if backend is None:  # fallback
		return cdf, pdf, ppf
	elif backend == 'mpmath':
		try:
			import mpmath
		except ImportError:
			raise ImportError('Install "mpmath" to use this backend')
		return mpmath.ncdf, mpmath.npdf, _gen_ppf(mpmath.erfc, math=mpmath)
	elif backend == 'scipy':
		try:
			from scipy.stats import norm
		except ImportError:
			raise ImportError('Install "scipy" to use this backend')
		return norm.cdf, norm.pdf, norm.ppf
	raise ValueError('%r backend is not defined' % backend)

def available_backends():
	backends = [None]
	for backend in ['mpmath', 'scipy']:
		try:
			__import__(backend)
		except ImportError:
			continue
		backends.append(backend)
	return backends

class Node(object):

	pass

class Variable(Node, Gaussian):

	def __init__(self):
		self.messages = {}
		super(Variable, self).__init__()

	def set(self, val):
		delta = self.delta(val)
		self.pi, self.tau = val.pi, val.tau
		return delta

	def delta(self, other):
		pi_delta = abs(self.pi - other.pi)
		if pi_delta == inf:
			return 0.
		return max(abs(self.tau - other.tau), math.sqrt(pi_delta))

	def update_message(self, factor, pi=0, tau=0, message=None):
		message = message or Gaussian(pi=pi, tau=tau)
		old_message, self[factor] = self[factor], message
		return self.set(self / old_message * message)

	def update_value(self, factor, pi=0, tau=0, value=None):
		value = value or Gaussian(pi=pi, tau=tau)
		old_message = self[factor]
		self[factor] = value * old_message / self
		return self.set(value)

	def __getitem__(self, factor):
		return self.messages[factor]

	def __setitem__(self, factor, message):
		self.messages[factor] = message

	def __repr__(self):
		args = (type(self).__name__, super(Variable, self).__repr__(),
				len(self.messages), '' if len(self.messages) == 1 else 's')
		return '<%s %s with %d connection%s>' % args


class Factor(Node):

	def __init__(self, variables):
		self.vars = variables
		for var in variables:
			var[self] = Gaussian()

	def down(self):
		return 0

	def up(self):
		return 0

	@property
	def var(self):
		assert len(self.vars) == 1
		return self.vars[0]

	def __repr__(self):
		args = (type(self).__name__, len(self.vars),
				'' if len(self.vars) == 1 else 's')
		return '<%s with %d connection%s>' % args


class PriorFactor(Factor):

	def __init__(self, var, val, dynamic=0):
		super(PriorFactor, self).__init__([var])
		self.val = val
		self.dynamic = dynamic

	def down(self):
		sigma = math.sqrt(self.val.sigma ** 2 + self.dynamic ** 2)
		value = Gaussian(self.val.mu, sigma)
		return self.var.update_value(self, value=value)


class LikelihoodFactor(Factor):

	def __init__(self, mean_var, value_var, variance):
		super(LikelihoodFactor, self).__init__([mean_var, value_var])
		self.mean = mean_var
		self.value = value_var
		self.variance = variance

	def calc_a(self, var):
		return 1. / (1. + self.variance * var.pi)

	def down(self):
		# update value.
		msg = self.mean / self.mean[self]
		a = self.calc_a(msg)
		return self.value.update_message(self, a * msg.pi, a * msg.tau)

	def up(self):
		# update mean.
		msg = self.value / self.value[self]
		a = self.calc_a(msg)
		return self.mean.update_message(self, a * msg.pi, a * msg.tau)


class SumFactor(Factor):

	def __init__(self, sum_var, term_vars, coeffs):
		super(SumFactor, self).__init__([sum_var] + term_vars)
		self.sum = sum_var
		self.terms = term_vars
		self.coeffs = coeffs

	def down(self):
		vals = self.terms
		msgs = [var[self] for var in vals]
		return self.update(self.sum, vals, msgs, self.coeffs)

	def up(self, index=0):
		coeff = self.coeffs[index]
		coeffs = []
		for x, c in enumerate(self.coeffs):
			try:
				if x == index:
					coeffs.append(1. / coeff)
				else:
					coeffs.append(-c / coeff)
			except ZeroDivisionError:
				coeffs.append(0.)
		vals = self.terms[:]
		vals[index] = self.sum
		msgs = [var[self] for var in vals]
		return self.update(self.terms[index], vals, msgs, coeffs)

	def update(self, var, vals, msgs, coeffs):
		pi_inv = 0
		mu = 0
		for val, msg, coeff in zip(vals, msgs, coeffs):
			div = val / msg
			mu += coeff * div.mu
			if pi_inv == inf:
				continue
			try:
				# numpy.float64 handles floating-point error by different way.
				# For example, it can just warn RuntimeWarning on n/0 problem
				# instead of throwing ZeroDivisionError.  So div.pi, the
				# denominator has to be a built-in float.
				pi_inv += coeff ** 2 / float(div.pi)
			except ZeroDivisionError:
				pi_inv = inf
		pi = 1. / pi_inv
		tau = pi * mu
		return var.update_message(self, pi, tau)


class TruncateFactor(Factor):

	def __init__(self, var, v_func, w_func, draw_margin):
		super(TruncateFactor, self).__init__([var])
		self.v_func = v_func
		self.w_func = w_func
		self.draw_margin = draw_margin

	def up(self):
		val = self.var
		msg = self.var[self]
		div = val / msg
		sqrt_pi = math.sqrt(div.pi)
		args = (div.tau / sqrt_pi, self.draw_margin * sqrt_pi)
		v = self.v_func(*args)
		w = self.w_func(*args)
		denom = (1. - w)
		pi, tau = div.pi / denom, (div.tau + sqrt_pi * v) / denom
		return val.update_value(self, pi, tau)

#: Default initial mean of ratings.
MU = 25.
#: Default initial standard deviation of ratings.
SIGMA = MU / 3
#: Default distance that guarantees about 76% chance of winning.
BETA = SIGMA / 2
#: Default dynamic factor.
TAU = SIGMA / 100
#: Default draw probability of the game.
DRAW_PROBABILITY = .10
#: A basis to check reliability of the result.
DELTA = 0.0001


def calc_draw_probability(draw_margin, size, env=None):
	if env is None:
		env = global_env()
	return 2 * env.cdf(draw_margin / (math.sqrt(size) * env.beta)) - 1


def calc_draw_margin(draw_probability, size, env=None):
	if env is None:
		env = global_env()
	return env.ppf((draw_probability + 1) / 2.) * math.sqrt(size) * env.beta


def _team_sizes(rating_groups):
	team_sizes = [0]
	for group in rating_groups:
		team_sizes.append(len(group) + team_sizes[-1])
	del team_sizes[0]
	return team_sizes


def _floating_point_error(env):
	if env.backend == 'mpmath':
		msg = 'Set "mpmath.mp.dps" to higher'
	else:
		msg = 'Cannot calculate correctly, set backend to "mpmath"'
	return FloatingPointError(msg)


class Rating(Gaussian):
	def __init__(self, mu=None, sigma=None):
		if isinstance(mu, tuple):
			mu, sigma = mu
		elif isinstance(mu, Gaussian):
			mu, sigma = mu.mu, mu.sigma
		if mu is None:
			mu = global_env().mu
		if sigma is None:
			sigma = global_env().sigma
		super(Rating, self).__init__(mu, sigma)

	def __int__(self):
		return int(self.mu)

	def __long__(self):
		return long(self.mu)

	def __float__(self):
		return float(self.mu)

	def __iter__(self):
		return iter((self.mu, self.sigma))

	def __repr__(self):
		c = type(self)
		args = ('.'.join([c.__module__, c.__name__]), self.mu, self.sigma)
		return '%s(mu=%.3f, sigma=%.3f)' % args


class TrueSkill(object):
	def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, tau=TAU,
				 draw_probability=DRAW_PROBABILITY, backend=None):
		self.mu = mu
		self.sigma = sigma
		self.beta = beta
		self.tau = tau
		self.draw_probability = draw_probability
		self.backend = backend
		if isinstance(backend, tuple):
			self.cdf, self.pdf, self.ppf = backend
		else:
			self.cdf, self.pdf, self.ppf = choose_backend(backend)

	def create_rating(self, mu=None, sigma=None):
		if mu is None:
			mu = self.mu
		if sigma is None:
			sigma = self.sigma
		return Rating(mu, sigma)

	def v_win(self, diff, draw_margin):
		x = diff - draw_margin
		denom = self.cdf(x)
		return (self.pdf(x) / denom) if denom else -x

	def v_draw(self, diff, draw_margin):
		abs_diff = abs(diff)
		a, b = draw_margin - abs_diff, -draw_margin - abs_diff
		denom = self.cdf(a) - self.cdf(b)
		numer = self.pdf(b) - self.pdf(a)
		return ((numer / denom) if denom else a) * (-1 if diff < 0 else +1)

	def w_win(self, diff, draw_margin):
		x = diff - draw_margin
		v = self.v_win(diff, draw_margin)
		w = v * (v + x)
		if 0 < w < 1:
			return w
		raise _floating_point_error(self)

	def w_draw(self, diff, draw_margin):
		abs_diff = abs(diff)
		a, b = draw_margin - abs_diff, -draw_margin - abs_diff
		denom = self.cdf(a) - self.cdf(b)
		if not denom:
			raise _floating_point_error(self)
		v = self.v_draw(abs_diff, draw_margin)
		return (v ** 2) + (a * self.pdf(a) - b * self.pdf(b)) / denom

	def validate_rating_groups(self, rating_groups):
		# check group sizes
		if len(rating_groups) < 2:
			raise ValueError('Need multiple rating groups')
		elif not all(rating_groups):
			raise ValueError('Each group must contain multiple ratings')
		# check group types
		group_types = set(map(type, rating_groups))
		if len(group_types) != 1:
			raise TypeError('All groups should be same type')
		elif group_types.pop() is Rating:
			raise TypeError('Rating cannot be a rating group')
		# normalize rating_groups
		if isinstance(rating_groups[0], dict):
			dict_rating_groups = rating_groups
			rating_groups = []
			keys = []
			for dict_rating_group in dict_rating_groups:
				rating_group, key_group = [], []
				for key, rating in iteritems(dict_rating_group):
					rating_group.append(rating)
					key_group.append(key)
				rating_groups.append(tuple(rating_group))
				keys.append(tuple(key_group))
		else:
			rating_groups = list(rating_groups)
			keys = None
		return rating_groups, keys

	def validate_weights(self, weights, rating_groups, keys=None):
		if weights is None:
			weights = [(1,) * len(g) for g in rating_groups]
		elif isinstance(weights, dict):
			weights_dict, weights = weights, []
			for x, group in enumerate(rating_groups):
				w = []
				weights.append(w)
				for y, rating in enumerate(group):
					if keys is not None:
						y = keys[x][y]
					w.append(weights_dict.get((x, y), 1))
		return weights

	def factor_graph_builders(self, rating_groups, ranks, weights):
		flatten_ratings = sum(map(tuple, rating_groups), ())
		flatten_weights = sum(map(tuple, weights), ())
		size = len(flatten_ratings)
		group_size = len(rating_groups)
		# create variables
		rating_vars = [Variable() for x in range(size)]
		perf_vars = [Variable() for x in range(size)]
		team_perf_vars = [Variable() for x in range(group_size)]
		team_diff_vars = [Variable() for x in range(group_size - 1)]
		team_sizes = _team_sizes(rating_groups)
		# layer builders
		def build_rating_layer():
			for rating_var, rating in zip(rating_vars, flatten_ratings):
				yield PriorFactor(rating_var, rating, self.tau)
		def build_perf_layer():
			for rating_var, perf_var in zip(rating_vars, perf_vars):
				yield LikelihoodFactor(rating_var, perf_var, self.beta ** 2)
		def build_team_perf_layer():
			for team, team_perf_var in enumerate(team_perf_vars):
				if team > 0:
					start = team_sizes[team - 1]
				else:
					start = 0
				end = team_sizes[team]
				child_perf_vars = perf_vars[start:end]
				coeffs = flatten_weights[start:end]
				yield SumFactor(team_perf_var, child_perf_vars, coeffs)
		def build_team_diff_layer():
			for team, team_diff_var in enumerate(team_diff_vars):
				yield SumFactor(team_diff_var,
								team_perf_vars[team:team + 2], [+1, -1])
		def build_trunc_layer():
			for x, team_diff_var in enumerate(team_diff_vars):
				if callable(self.draw_probability):
					# dynamic draw probability
					team_perf1, team_perf2 = team_perf_vars[x:x + 2]
					args = (Rating(team_perf1), Rating(team_perf2), self)
					draw_probability = self.draw_probability(*args)
				else:
					# static draw probability
					draw_probability = self.draw_probability
				size = sum(map(len, rating_groups[x:x + 2]))
				draw_margin = calc_draw_margin(draw_probability, size, self)
				if ranks[x] == ranks[x + 1]:  # is a tie?
					v_func, w_func = self.v_draw, self.w_draw
				else:
					v_func, w_func = self.v_win, self.w_win
				yield TruncateFactor(team_diff_var,
									 v_func, w_func, draw_margin)
		# build layers
		return (build_rating_layer, build_perf_layer, build_team_perf_layer,
				build_team_diff_layer, build_trunc_layer)

	def run_schedule(self, build_rating_layer, build_perf_layer,
					 build_team_perf_layer, build_team_diff_layer,
					 build_trunc_layer, min_delta=DELTA):
		if min_delta <= 0:
			raise ValueError('min_delta must be greater than 0')
		layers = []
		def build(builders):
			layers_built = [list(build()) for build in builders]
			layers.extend(layers_built)
			return layers_built
		# gray arrows
		layers_built = build([build_rating_layer,
							  build_perf_layer,
							  build_team_perf_layer])
		rating_layer, perf_layer, team_perf_layer = layers_built
		for f in chain(*layers_built):
			f.down()
		# arrow #1, #2, #3
		team_diff_layer, trunc_layer = build([build_team_diff_layer,
											  build_trunc_layer])
		team_diff_len = len(team_diff_layer)
		for x in range(10):
			if team_diff_len == 1:
				# only two teams
				team_diff_layer[0].down()
				delta = trunc_layer[0].up()
			else:
				# multiple teams
				delta = 0
				for x in range(team_diff_len - 1):
					team_diff_layer[x].down()
					delta = max(delta, trunc_layer[x].up())
					team_diff_layer[x].up(1)  # up to right variable
				for x in range(team_diff_len - 1, 0, -1):
					team_diff_layer[x].down()
					delta = max(delta, trunc_layer[x].up())
					team_diff_layer[x].up(0)  # up to left variable
			# repeat until to small update
			if delta <= min_delta:
				break
		# up both ends
		team_diff_layer[0].up(0)
		team_diff_layer[team_diff_len - 1].up(1)
		# up the remainder of the black arrows
		for f in team_perf_layer:
			for x in range(len(f.vars) - 1):
				f.up(x)
		for f in perf_layer:
			f.up()
		return layers

	def rate(self, rating_groups, ranks=None, weights=None, min_delta=DELTA):
		rating_groups, keys = self.validate_rating_groups(rating_groups)
		weights = self.validate_weights(weights, rating_groups, keys)
		group_size = len(rating_groups)
		if ranks is None:
			ranks = range(group_size)
		elif len(ranks) != group_size:
			raise ValueError('Wrong ranks')
		# sort rating groups by rank
		by_rank = lambda x: x[1][1]
		sorting = sorted(enumerate(zip(rating_groups, ranks, weights)),
						 key=by_rank)
		sorted_rating_groups, sorted_ranks, sorted_weights = [], [], []
		for x, (g, r, w) in sorting:
			sorted_rating_groups.append(g)
			sorted_ranks.append(r)
			# make weights to be greater than 0
			sorted_weights.append(max(min_delta, w_) for w_ in w)
		# build factor graph
		args = (sorted_rating_groups, sorted_ranks, sorted_weights)
		builders = self.factor_graph_builders(*args)
		args = builders + (min_delta,)
		layers = self.run_schedule(*args)
		# make result
		rating_layer, team_sizes = layers[0], _team_sizes(sorted_rating_groups)
		transformed_groups = []
		for start, end in zip([0] + team_sizes[:-1], team_sizes):
			group = []
			for f in rating_layer[start:end]:
				group.append(Rating(float(f.var.mu), float(f.var.sigma)))
			transformed_groups.append(tuple(group))
		by_hint = lambda x: x[0]
		unsorting = sorted(zip((x for x, __ in sorting), transformed_groups),
						   key=by_hint)
		if keys is None:
			return [g for x, g in unsorting]
		# restore the structure with input dictionary keys
		return [dict(zip(keys[x], g)) for x, g in unsorting]

	def quality(self, rating_groups, weights=None):
		rating_groups, keys = self.validate_rating_groups(rating_groups)
		weights = self.validate_weights(weights, rating_groups, keys)
		flatten_ratings = sum(map(tuple, rating_groups), ())
		flatten_weights = sum(map(tuple, weights), ())
		length = len(flatten_ratings)
		# a vector of all of the skill means
		mean_matrix = Matrix([[r.mu] for r in flatten_ratings])
		# a matrix whose diagonal values are the variances (sigma ** 2) of each
		# of the players.
		def variance_matrix(height, width):
			variances = (r.sigma ** 2 for r in flatten_ratings)
			for x, variance in enumerate(variances):
				yield (x, x), variance
		variance_matrix = Matrix(variance_matrix, length, length)
		# the player-team assignment and comparison matrix
		def rotated_a_matrix(set_height, set_width):
			t = 0
			for r, (cur, _next) in enumerate(zip(rating_groups[:-1],
												rating_groups[1:])):
				for x in range(t, t + len(cur)):
					yield (r, x), flatten_weights[x]
					t += 1
				x += 1
				for x in range(x, x + len(_next)):
					yield (r, x), -flatten_weights[x]
			set_height(r + 1)
			set_width(x + 1)
		rotated_a_matrix = Matrix(rotated_a_matrix)
		a_matrix = rotated_a_matrix.transpose()
		# match quality further derivation
		_ata = (self.beta ** 2) * rotated_a_matrix * a_matrix
		_atsa = rotated_a_matrix * variance_matrix * a_matrix
		start = mean_matrix.transpose() * a_matrix
		middle = _ata + _atsa
		end = rotated_a_matrix * mean_matrix
		# make result
		e_arg = (-0.5 * start * middle.inverse() * end).determinant()
		s_arg = _ata.determinant() / middle.determinant()
		return math.exp(e_arg) * math.sqrt(s_arg)

	def expose(self, rating):
		k = self.mu / self.sigma
		return rating.mu - k * rating.sigma

	def make_as_global(self):
		return setup(env=self)

	def __repr__(self):
		c = type(self)
		if callable(self.draw_probability):
			f = self.draw_probability
			draw_probability = '.'.join([f.__module__, f.__name__])
		else:
			draw_probability = '%.1f%%' % (self.draw_probability * 100)
		if self.backend is None:
			backend = ''
		elif isinstance(self.backend, tuple):
			backend = ', backend=...'
		else:
			backend = ', backend=%r' % self.backend
		args = ('.'.join([c.__module__, c.__name__]), self.mu, self.sigma,
				self.beta, self.tau, draw_probability, backend)
		return ('%s(mu=%.3f, sigma=%.3f, beta=%.3f, tau=%.3f, '
				'draw_probability=%s%s)' % args)


def rate_1vs1(rating1, rating2, drawn=False, min_delta=DELTA, env=None):
	if env is None:
		env = global_env()
	ranks = [0, 0 if drawn else 1]
	teams = env.rate([(rating1,), (rating2,)], ranks, min_delta=min_delta)
	return teams[0][0], teams[1][0]


def quality_1vs1(rating1, rating2, env=None):
	if env is None:
		env = global_env()
	return env.quality([(rating1,), (rating2,)])


def global_env():
	try:
		global_env.__trueskill__
	except AttributeError:
		# setup the default environment
		setup()
	return global_env.__trueskill__


def setup(mu=MU, sigma=SIGMA, beta=BETA, tau=TAU,
		  draw_probability=DRAW_PROBABILITY, backend=None, env=None):
	if env is None:
		env = TrueSkill(mu, sigma, beta, tau, draw_probability, backend)
	global_env.__trueskill__ = env
	return env


def rate(rating_groups, ranks=None, weights=None, min_delta=DELTA):
	return global_env().rate(rating_groups, ranks, weights, min_delta)


def quality(rating_groups, weights=None):
	return global_env().quality(rating_groups, weights)


def expose(rating):
	return global_env().expose(rating)