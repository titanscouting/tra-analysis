import numpy as np

from tra_analysis import Analysis as an
from tra_analysis import Array
from tra_analysis import ClassificationMetric
from tra_analysis import CorrelationTest
from tra_analysis import Fit
from tra_analysis import KNN
from tra_analysis import NaiveBayes
from tra_analysis import RandomForest
from tra_analysis import RegressionMetric
from tra_analysis import Sort
from tra_analysis import StatisticalTest
from tra_analysis import SVM

def test_():

	test_data_linear = [1, 3, 6, 7, 9]
	test_data_linear2 = [2, 2, 5, 7, 13]
	test_data_array = Array(test_data_linear)
	
	x_data_circular = []
	y_data_circular = []
	
	y_data_ccu = [1, 3, 7, 14, 21]
	y_data_ccd = [1, 5, 7, 8.5, 8.66]
	
	test_data_scrambled = [-32, 34, 19, 72, -65, -11, -43, 6, 85, -17, -98, -26, 12, 20, 9, -92, -40, 98, -78, 17, -20, 49, 93, -27, -24, -66, 40, 84, 1, -64, -68, -25, -42, -46, -76, 43, -3, 30, -14, -34, -55, -13, 41, -30, 0, -61, 48, 23, 60, 87, 80, 77, 53, 73, 79, 24, -52, 82, 8, -44, 65, 47, -77, 94, 7, 37, -79, 36, -94, 91, 59, 10, 97, -38, -67, 83, 54, 31, -95, -63, 16, -45, 21, -12, 66, -48, -18, -96, -90, -21, -83, -74, 39, 64, 69, -97, 13, 55, 27, -39]
	test_data_sorted = [-98, -97, -96, -95, -94, -92, -90, -83, -79, -78, -77, -76, -74, -68, -67, -66, -65, -64, -63, -61, -55, -52, -48, -46, -45, -44, -43, -42, -40, -39, -38, -34, -32, -30, -27, -26, -25, -24, -21, -20, -18, -17, -14, -13, -12, -11, -3, 0, 1, 6, 7, 8, 9, 10, 12, 13, 16, 17, 19, 20, 21, 23, 24, 27, 30, 31, 34, 36, 37, 39, 40, 41, 43, 47, 48, 49, 53, 54, 55, 59, 60, 64, 65, 66, 69, 72, 73, 77, 79, 80, 82, 83, 84, 85, 87, 91, 93, 94, 97, 98]
	
	test_data_2D_pairs = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
	test_labels_2D_pairs = np.array([1, 1, 2, 2])
	validation_data_2D_pairs = np.array([[-0.8, -1], [0.8, 1.2]])
	validation_labels_2D_pairs = np.array([1, 2])

	assert an.basic_stats(test_data_linear) == {"mean": 5.2, "median": 6.0, "standard-deviation": 2.85657137141714, "variance": 8.16, "minimum": 1.0, "maximum": 9.0}
	assert an.z_score(3.2, 6, 1.5) == -1.8666666666666665
	assert an.z_normalize([test_data_linear], 1).tolist() == [[0.07537783614444091, 0.22613350843332272, 0.45226701686664544, 0.5276448530110863, 0.6784005252999682]]
	assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["lin"])) == True
	#assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccd, ["log"])) == True
	#assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["exp"])) == True
	#assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["ply"])) == True
	#assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccd, ["sig"])) == True
	assert an.Metric().elo(1500, 1500, [1, 0], 400, 24) == 1512.0
	assert an.Metric().glicko2(1500, 250, 0.06, [1500, 1400], [250, 240], [1, 0]) == (1478.864307445517, 195.99122679202452, 0.05999602937563585)
	#assert an.Metric().trueskill([[(25, 8.33), (24, 8.25), (32, 7.5)], [(25, 8.33), (25, 8.33), (21, 6.5)]], [1, 0]) == [(metrics.trueskill.Rating(mu=21.346, sigma=7.875), metrics.trueskill.Rating(mu=20.415, sigma=7.808), metrics.trueskill.Rating(mu=29.037, sigma=7.170)), (metrics.trueskill.Rating(mu=28.654, sigma=7.875), metrics.trueskill.Rating(mu=28.654, sigma=7.875), metrics.trueskill.Rating(mu=23.225, sigma=6.287))]

	assert test_data_array.elementwise_mean() == 5.2
	assert test_data_array.elementwise_median() == 6.0
	assert test_data_array.elementwise_stdev() == 2.85657137141714
	assert test_data_array.elementwise_variance() == 8.16
	assert test_data_array.elementwise_npmin() == 1
	assert test_data_array.elementwise_npmax() == 9
	assert test_data_array.elementwise_stats() == (5.2, 6.0, 2.85657137141714, 8.16, 1, 9)

	assert CorrelationTest.anova_oneway(test_data_linear, test_data_linear2) == {"f-value": 0.05825242718446602, "p-value": 0.8153507906592907}
	assert CorrelationTest.pearson(test_data_linear, test_data_linear2) == {"r-value":0.9153061540753286, "p-value": 0.02920895440940874}
	assert CorrelationTest.spearman(test_data_linear, test_data_linear2) == {"r-value":0.9746794344808964, "p-value":0.004818230468198537}
	assert CorrelationTest.point_biserial(test_data_linear, test_data_linear2) == {"r-value":0.9153061540753286, "p-value":0.02920895440940874}
	assert CorrelationTest.kendall(test_data_linear, test_data_linear2) == {"tau":0.9486832980505137, "p-value":0.022977401503206086}
	assert CorrelationTest.kendall_weighted(test_data_linear, test_data_linear2) == {"tau":0.9750538072369643, "p-value":np.nan}
	#assert CorrelationTest.mgc()

	assert all(a == b for a, b in zip(Sort.quicksort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.mergesort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.heapsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.introsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.insertionsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.timsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.selectionsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.shellsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.bubblesort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.cyclesort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(Sort.cocktailsort(test_data_scrambled), test_data_sorted))
	
	assert Fit.CircleFit(x=[0,0,-1,1], y=[1, -1, 0, 0]).LSC() == (0.0, 0.0, 1.0, 0.0)

	SVM(test_data_2D_pairs, test_labels_2D_pairs, validation_data_2D_pairs, validation_labels_2D_pairs)

def SVM(data, labels, test_data, test_labels):

	lin_kernel = SVM.PrebuiltKernel.Linear()
	ply_kernel = SVM.PrebuiltKernel.Polynomial()
	rbf_kernel = SVM.PrebuiltKernel.RBF()
	sig_kernel = SVM.PrebuiltKernel.Sigmoid()

	lin_kernel = SVM.fit(lin_kernel, data, labels)
	ply_kernel = SVM.fit(ply_kernel, data, labels)
	rbf_kernel = SVM.fit(rbf_kernel, data, labels)
	sig_kernel = SVM.fit(sig_kernel, data, labels)

	for i in range(test_data):

		assert lin_kernel.predict([test_data[i]]).tolist() == [test[i]]

	for i in range(test_data):

		assert ply_kernel.predict([test_data[i]]).tolist() == [test[i]]

	for i in range(test_data):

		assert rbf_kernel.predict([test_data[i]]).tolist() == [test[i]]

	for i in range(test_data):

		assert sig_kernel.predict([test_data[i]]).tolist() == [test[i]]