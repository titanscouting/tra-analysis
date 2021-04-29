# Only included for backwards compatibility! Do not update, CorrelationTest is preferred and supported.

import scipy
from scipy import stats

class CorrelationTest:

	def anova_oneway(self, *args): #expects arrays of samples

		results = scipy.stats.f_oneway(*args)
		return {"f-value": results[0], "p-value": results[1]}

	def pearson(self, x, y):

		results = scipy.stats.pearsonr(x, y)
		return {"r-value": results[0], "p-value": results[1]}

	def spearman(self, a, b = None, axis = 0, nan_policy = 'propagate'):

		results = scipy.stats.spearmanr(a, b = b, axis = axis, nan_policy = nan_policy)
		return {"r-value": results[0], "p-value": results[1]}

	def point_biserial(self, x,y):

		results = scipy.stats.pointbiserialr(x, y)
		return {"r-value": results[0], "p-value": results[1]}

	def kendall(self, x, y, initial_lexsort = None, nan_policy = 'propagate', method = 'auto'):

		results = scipy.stats.kendalltau(x, y, initial_lexsort = initial_lexsort, nan_policy = nan_policy, method = method)
		return {"tau": results[0], "p-value": results[1]}

	def kendall_weighted(self, x, y, rank = True, weigher = None, additive = True):

		results = scipy.stats.weightedtau(x, y, rank = rank, weigher = weigher, additive = additive)
		return {"tau": results[0], "p-value": results[1]}

	def mgc(self, x, y, compute_distance = None, reps = 1000, workers = 1, is_twosamp = False, random_state = None):

		results = scipy.stats.multiscale_graphcorr(x, y, compute_distance = compute_distance, reps = reps, workers = workers, is_twosamp = is_twosamp, random_state = random_state)
		return {"k-value": results[0], "p-value": results[1], "data": results[2]} # unsure if MGC test returns a k value