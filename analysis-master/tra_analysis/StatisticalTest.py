# Titan Robotics Team 2022: StatisticalTest submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import StatisticalTest'
# setup:

__version__ = "1.0.3"

__changelog__ = """changelog:
	1.0.3:
		- optimized imports
	1.0.2:
		- added tukey_multicomparison
		- fixed styling
	1.0.1:
		- fixed typo in __all__
	1.0.0:
		- ported analysis.StatisticalTest() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
	"James Pan <zpan@imsa.edu>",
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
	'normaltest',
	'tukey_multicomparison'
]

import numpy as np
import scipy

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

def get_tukeyQcrit(k, df, alpha=0.05):
	'''
	From statsmodels.sandbox.stats.multicomp
	
	return critical values for Tukey's HSD (Q)

	Parameters
	----------
	k : int in {2, ..., 10}
		number of tests
	df : int
		degrees of freedom of error term
	alpha : {0.05, 0.01}
		type 1 error, 1-confidence level

	not enough error checking for limitations
	'''
	# qtable from statsmodels.sandbox.stats.multicomp
	qcrit = '''
	2     3     4     5     6     7     8     9     10
	5   3.64 5.70   4.60 6.98   5.22 7.80   5.67 8.42   6.03 8.91   6.33 9.32   6.58 9.67   6.80 9.97   6.99 10.24
	6   3.46 5.24   4.34 6.33   4.90 7.03   5.30 7.56   5.63 7.97   5.90 8.32   6.12 8.61   6.32 8.87   6.49 9.10
	7   3.34 4.95   4.16 5.92   4.68 6.54   5.06 7.01   5.36 7.37   5.61 7.68   5.82 7.94   6.00 8.17   6.16 8.37
	8   3.26 4.75   4.04 5.64   4.53 6.20   4.89 6.62   5.17 6.96   5.40 7.24       5.60 7.47   5.77 7.68   5.92 7.86
	9   3.20 4.60   3.95 5.43   4.41 5.96   4.76 6.35   5.02 6.66   5.24 6.91       5.43 7.13   5.59 7.33   5.74 7.49
	10  3.15 4.48   3.88 5.27   4.33 5.77   4.65 6.14   4.91 6.43   5.12 6.67       5.30 6.87   5.46 7.05   5.60 7.21
	11  3.11 4.39   3.82 5.15   4.26 5.62   4.57 5.97   4.82 6.25   5.03 6.48 5.20 6.67   5.35 6.84   5.49 6.99
	12  3.08 4.32   3.77 5.05   4.20 5.50   4.51 5.84   4.75 6.10   4.95 6.32 5.12 6.51   5.27 6.67   5.39 6.81
	13  3.06 4.26   3.73 4.96   4.15 5.40   4.45 5.73   4.69 5.98   4.88 6.19 5.05 6.37   5.19 6.53   5.32 6.67
	14  3.03 4.21   3.70 4.89   4.11 5.32   4.41 5.63   4.64 5.88   4.83 6.08 4.99 6.26   5.13 6.41   5.25 6.54
	15  3.01 4.17   3.67 4.84   4.08 5.25   4.37 5.56   4.59 5.80   4.78 5.99 4.94 6.16   5.08 6.31   5.20 6.44
	16  3.00 4.13   3.65 4.79   4.05 5.19   4.33 5.49   4.56 5.72   4.74 5.92 4.90 6.08   5.03 6.22   5.15 6.35
	17  2.98 4.10   3.63 4.74   4.02 5.14   4.30 5.43   4.52 5.66   4.70 5.85 4.86 6.01   4.99 6.15   5.11 6.27
	18  2.97 4.07   3.61 4.70   4.00 5.09   4.28 5.38   4.49 5.60   4.67 5.79 4.82 5.94   4.96 6.08   5.07 6.20
	19  2.96 4.05   3.59 4.67   3.98 5.05   4.25 5.33   4.47 5.55   4.65 5.73 4.79 5.89   4.92 6.02   5.04 6.14
	20  2.95 4.02   3.58 4.64   3.96 5.02   4.23 5.29   4.45 5.51   4.62 5.69 4.77 5.84   4.90 5.97   5.01 6.09
	24  2.92 3.96   3.53 4.55   3.90 4.91   4.17 5.17   4.37 5.37   4.54 5.54 4.68 5.69   4.81 5.81   4.92 5.92
	30  2.89 3.89   3.49 4.45   3.85 4.80   4.10 5.05   4.30 5.24   4.46 5.40 4.60 5.54   4.72 5.65   4.82 5.76
	40  2.86 3.82   3.44 4.37   3.79 4.70   4.04 4.93   4.23 5.11   4.39 5.26 4.52 5.39   4.63 5.50   4.73 5.60
	60  2.83 3.76   3.40 4.28   3.74 4.59   3.98 4.82   4.16 4.99   4.31 5.13 4.44 5.25   4.55 5.36   4.65 5.45
	120   2.80 3.70   3.36 4.20   3.68 4.50   3.92 4.71   4.10 4.87   4.24 5.01 4.36 5.12   4.47 5.21   4.56 5.30
	infinity  2.77 3.64   3.31 4.12   3.63 4.40   3.86 4.60   4.03 4.76   4.17 4.88   4.29 4.99   4.39 5.08   4.47 5.16
	'''
	res = [line.split() for line in qcrit.replace('infinity','9999').split('\n')]
	c=np.array(res[2:-1]).astype(float)
	#c[c==9999] = np.inf
	ccols = np.arange(2,11)
	crows = c[:,0]
	cv005 = c[:, 1::2]
	cv001 = c[:, 2::2]

	if alpha == 0.05:
		intp = scipy.interpolate.interp1d(crows, cv005[:,k-2])
	elif alpha == 0.01:
		intp = scipy.interpolate.interp1d(crows, cv001[:,k-2])
	else:
		raise ValueError('only implemented for alpha equal to 0.01 and 0.05')
	return intp(df)

def tukey_multicomparison(groups, alpha=0.05):
	#formulas according to https://astatsa.com/OneWay_Anova_with_TukeyHSD/

	k = len(groups)
	df = 0
	means = []
	MSE = 0
	for group in groups:
		df+= len(group)
		mean = sum(group)/len(group)
		means.append(mean)
		MSE += sum([(i-mean)**2 for i in group])
	df -= k
	MSE /= df

	q_dict = {}
	crit_q = get_tukeyQcrit(k, df, alpha)

	for i in range(k-1):
		for j in range(i+1, k):
			numerator = abs(means[i] - means[j])
			denominator = np.sqrt( MSE / ( 2/(1/len(groups[i]) + 1/len(groups[j])) ))
			q = numerator/denominator
			q_dict["group "+ str(i+1) + " and group " + str(j+1)] = [q, q>crit_q]
	
	return q_dict