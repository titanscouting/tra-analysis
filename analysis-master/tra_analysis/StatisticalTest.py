# Titan Robotics Team 2022: StatisticalTest submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import StatisticalTest'
# setup:

__version__ = "1.0.1"

__changelog__ = """changelog:
	1.0.1:
		- fixed typo in __all__
	1.0.0:
		- ported analysis.StatisticalTest() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = [
	'ttest_onesample',
	'ttest_independent',
	'ttest_statistic',
	'ttest_related',
	'ks_fitness',
	'chisquare',
	'powerdivergence'
	'ks_twosample',
	'es_twosample',
	'mw_rank',
	'mw_tiecorrection',
	'rankdata',
	'wilcoxon_ranksum',
	'wilcoxon_signedrank',
	'kw_htest',
	'friedman_chisquare',
	'bm_wtest',
	'combine_pvalues',
	'jb_fitness',
	'ab_equality',
	'bartlett_variance',
	'levene_variance',
	'sw_normality',
	'shapiro',
	'ad_onesample',
	'ad_ksample',
	'binomial',
	'fk_variance',
	'mood_mediantest',
	'mood_equalscale',
	'skewtest',
	'kurtosistest',
	'normaltest'
]

import scipy
from scipy import stats

def ttest_onesample(a, popmean, axis = 0, nan_policy = 'propagate'):

	results = scipy.stats.ttest_1samp(a, popmean, axis = axis, nan_policy = nan_policy)
	return {"t-value": results[0], "p-value": results[1]}

def ttest_independent(a, b, equal = True, nan_policy = 'propagate'):

	results = scipy.stats.ttest_ind(a, b, equal_var = equal, nan_policy = nan_policy)
	return {"t-value": results[0], "p-value": results[1]}

def ttest_statistic(o1, o2, equal = True):

	results = scipy.stats.ttest_ind_from_stats(o1["mean"], o1["std"], o1["nobs"], o2["mean"], o2["std"], o2["nobs"], equal_var = equal)
	return {"t-value": results[0], "p-value": results[1]}

def ttest_related(a, b, axis = 0, nan_policy='propagate'):

	results = scipy.stats.ttest_rel(a, b, axis = axis, nan_policy = nan_policy)
	return {"t-value": results[0], "p-value": results[1]}

def ks_fitness(rvs, cdf, args = (), N = 20, alternative = 'two-sided', mode = 'approx'):

	results = scipy.stats.kstest(rvs, cdf, args = args, N = N, alternative = alternative, mode = mode)
	return {"ks-value": results[0], "p-value": results[1]}

def chisquare(f_obs, f_exp = None, ddof = None, axis = 0):

	results = scipy.stats.chisquare(f_obs, f_exp = f_exp, ddof = ddof, axis = axis)
	return {"chisquared-value": results[0], "p-value": results[1]}

def powerdivergence(f_obs, f_exp = None, ddof = None, axis = 0, lambda_ = None):

	results = scipy.stats.power_divergence(f_obs, f_exp = f_exp, ddof = ddof, axis = axis, lambda_ = lambda_)
	return {"powerdivergence-value": results[0], "p-value": results[1]}

def ks_twosample(x, y, alternative = 'two_sided', mode = 'auto'):
	
	results = scipy.stats.ks_2samp(x, y, alternative = alternative, mode = mode)
	return {"ks-value": results[0], "p-value": results[1]}

def es_twosample(x, y, t = (0.4, 0.8)):

	results = scipy.stats.epps_singleton_2samp(x, y, t = t)
	return {"es-value": results[0], "p-value": results[1]}

def mw_rank(x, y, use_continuity = True, alternative = None):

	results = scipy.stats.mannwhitneyu(x, y, use_continuity = use_continuity, alternative = alternative)
	return {"u-value": results[0], "p-value": results[1]}

def mw_tiecorrection(rank_values):

	results = scipy.stats.tiecorrect(rank_values)
	return {"correction-factor": results}

def rankdata(a, method = 'average'):

	results = scipy.stats.rankdata(a, method = method)
	return results

def wilcoxon_ranksum(a, b): # this seems to be superceded by Mann Whitney Wilcoxon U Test

	results = scipy.stats.ranksums(a, b)
	return {"u-value": results[0], "p-value": results[1]}

def wilcoxon_signedrank(x, y = None, zero_method = 'wilcox', correction = False, alternative = 'two-sided'):

	results = scipy.stats.wilcoxon(x, y = y, zero_method = zero_method, correction = correction, alternative = alternative)
	return {"t-value": results[0], "p-value": results[1]}

def kw_htest(*args, nan_policy = 'propagate'):

	results = scipy.stats.kruskal(*args, nan_policy = nan_policy)
	return {"h-value": results[0], "p-value": results[1]}

def friedman_chisquare(*args):

	results = scipy.stats.friedmanchisquare(*args)
	return {"chisquared-value": results[0], "p-value": results[1]}

def bm_wtest(x, y, alternative = 'two-sided', distribution = 't', nan_policy = 'propagate'):

	results = scipy.stats.brunnermunzel(x, y, alternative = alternative, distribution = distribution, nan_policy = nan_policy)
	return {"w-value": results[0], "p-value": results[1]}

def combine_pvalues(pvalues, method = 'fisher', weights = None):

	results = scipy.stats.combine_pvalues(pvalues, method = method, weights = weights)
	return {"combined-statistic": results[0], "p-value": results[1]}

def jb_fitness(x):

	results = scipy.stats.jarque_bera(x)
	return {"jb-value": results[0], "p-value": results[1]}

def ab_equality(x, y):

	results = scipy.stats.ansari(x, y)
	return {"ab-value": results[0], "p-value": results[1]}

def bartlett_variance(*args):

	results = scipy.stats.bartlett(*args)
	return {"t-value": results[0], "p-value": results[1]}

def levene_variance(*args, center = 'median', proportiontocut = 0.05):

	results = scipy.stats.levene(*args, center = center, proportiontocut = proportiontocut)
	return {"w-value": results[0], "p-value": results[1]}

def sw_normality(x):

	results = scipy.stats.shapiro(x)
	return {"w-value": results[0], "p-value": results[1]}

def shapiro(x):

	return "destroyed by facts and logic"

def ad_onesample(x, dist = 'norm'):

	results = scipy.stats.anderson(x, dist = dist)
	return {"d-value": results[0], "critical-values": results[1], "significance-value": results[2]}

def ad_ksample(samples, midrank = True):

	results = scipy.stats.anderson_ksamp(samples, midrank = midrank)
	return {"d-value": results[0], "critical-values": results[1], "significance-value": results[2]}

def binomial(x, n = None, p = 0.5, alternative = 'two-sided'):

	results = scipy.stats.binom_test(x, n = n, p = p, alternative = alternative)
	return {"p-value": results}

def fk_variance(*args, center = 'median', proportiontocut = 0.05):

	results = scipy.stats.fligner(*args, center = center, proportiontocut = proportiontocut)
	return {"h-value": results[0], "p-value": results[1]} # unknown if the statistic is an h value

def mood_mediantest(*args, ties = 'below', correction = True, lambda_ = 1, nan_policy = 'propagate'):

	results = scipy.stats.median_test(*args, ties = ties, correction = correction, lambda_ = lambda_, nan_policy = nan_policy)
	return {"chisquared-value": results[0], "p-value": results[1], "m-value": results[2], "table": results[3]}

def mood_equalscale(x, y, axis = 0):

	results = scipy.stats.mood(x, y, axis = axis)
	return {"z-score": results[0], "p-value": results[1]}

def skewtest(a, axis = 0, nan_policy = 'propogate'):

	results = scipy.stats.skewtest(a, axis = axis, nan_policy = nan_policy)
	return {"z-score": results[0], "p-value": results[1]}

def kurtosistest(a, axis = 0, nan_policy = 'propogate'):

	results = scipy.stats.kurtosistest(a, axis = axis, nan_policy = nan_policy)
	return {"z-score": results[0], "p-value": results[1]}

def normaltest(a, axis = 0, nan_policy = 'propogate'):

	results = scipy.stats.normaltest(a, axis = axis, nan_policy = nan_policy)
	return {"z-score": results[0], "p-value": results[1]}