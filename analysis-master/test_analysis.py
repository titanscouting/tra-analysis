import numpy as np
import sklearn
from sklearn import metrics

from tra_analysis import Analysis as an
from tra_analysis import Array
from tra_analysis import ClassificationMetric
from tra_analysis import Clustering
from tra_analysis import CorrelationTest
from tra_analysis import Fit
from tra_analysis import KNN
from tra_analysis import metrics as m
from tra_analysis import NaiveBayes
from tra_analysis import RandomForest
from tra_analysis import RegressionMetric
from tra_analysis import Sort
from tra_analysis import StatisticalTest
from tra_analysis import SVM

from tra_analysis.equation.parser import BNF

test_data_linear = [1, 3, 6, 7, 9]
test_data_linear2 = [2, 2, 5, 7, 13]
test_data_linear3 = [2, 5, 8, 6, 14]
test_data_array = Array(test_data_linear)

x_data_circular = []
y_data_circular = []

y_data_ccu = [1, 3, 7, 14, 21]
y_data_ccd = [8.66, 8.5, 7, 5, 1]

test_data_scrambled = [-32, 34, 19, 72, -65, -11, -43, 6, 85, -17, -98, -26, 12, 20, 9, -92, -40, 98, -78, 17, -20, 49, 93, -27, -24, -66, 40, 84, 1, -64, -68, -25, -42, -46, -76, 43, -3, 30, -14, -34, -55, -13, 41, -30, 0, -61, 48, 23, 60, 87, 80, 77, 53, 73, 79, 24, -52, 82, 8, -44, 65, 47, -77, 94, 7, 37, -79, 36, -94, 91, 59, 10, 97, -38, -67, 83, 54, 31, -95, -63, 16, -45, 21, -12, 66, -48, -18, -96, -90, -21, -83, -74, 39, 64, 69, -97, 13, 55, 27, -39]
test_data_sorted = [-98, -97, -96, -95, -94, -92, -90, -83, -79, -78, -77, -76, -74, -68, -67, -66, -65, -64, -63, -61, -55, -52, -48, -46, -45, -44, -43, -42, -40, -39, -38, -34, -32, -30, -27, -26, -25, -24, -21, -20, -18, -17, -14, -13, -12, -11, -3, 0, 1, 6, 7, 8, 9, 10, 12, 13, 16, 17, 19, 20, 21, 23, 24, 27, 30, 31, 34, 36, 37, 39, 40, 41, 43, 47, 48, 49, 53, 54, 55, 59, 60, 64, 65, 66, 69, 72, 73, 77, 79, 80, 82, 83, 84, 85, 87, 91, 93, 94, 97, 98]

test_data_2D_pairs = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
test_data_2D_positive = np.array([[23, 51], [21, 32], [15, 25], [17, 31]])
test_output = np.array([1, 3, 4, 5])
test_labels_2D_pairs = np.array([1, 1, 2, 2])
validation_data_2D_pairs = np.array([[-0.8, -1], [0.8, 1.2]])
validation_labels_2D_pairs = np.array([1, 2])

def test_basicstats():

	assert an.basic_stats(test_data_linear) == {"mean": 5.2, "median": 6.0, "standard-deviation": 2.85657137141714, "variance": 8.16, "minimum": 1.0, "maximum": 9.0}
	assert an.z_score(3.2, 6, 1.5) == -1.8666666666666665
	assert an.z_normalize([test_data_linear], 1).tolist() == [[0.07537783614444091, 0.22613350843332272, 0.45226701686664544, 0.5276448530110863, 0.6784005252999682]]

def test_regression():

	assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["lin"])) == True
	assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccd, ["log"])) == True
	assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["exp"])) == True
	assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["ply"])) == True
	assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccd, ["sig"])) == True

def test_metrics():

	assert an.Metric().elo(1500, 1500, [1, 0], 400, 24) == 1512.0
	assert an.Metric().glicko2(1500, 250, 0.06, [1500, 1400], [250, 240], [1, 0]) == (1478.864307445517, 195.99122679202452, 0.05999602937563585)
	e = [[(21.346, 7.875), (20.415, 7.808), (29.037, 7.170)], [(28.654, 7.875), (28.654, 7.875), (23.225, 6.287)]]
	r = an.Metric().trueskill([[(25, 8.33), (24, 8.25), (32, 7.5)], [(25, 8.33), (25, 8.33), (21, 6.5)]], [1, 0])
	i = 0
	for group in r:
		j = 0
		for team in group:
			assert abs(team.mu - e[i][j][0]) < 0.001
			assert abs(team.sigma - e[i][j][1]) < 0.001
			j+=1
		i+=1

def test_array():

	assert test_data_array.elementwise_mean() == 5.2
	assert test_data_array.elementwise_median() == 6.0
	assert test_data_array.elementwise_stdev() == 2.85657137141714
	assert test_data_array.elementwise_variance() == 8.16
	assert test_data_array.elementwise_npmin() == 1
	assert test_data_array.elementwise_npmax() == 9
	assert test_data_array.elementwise_stats() == (5.2, 6.0, 2.85657137141714, 8.16, 1, 9)

	for i in range(len(test_data_array)):
		assert test_data_array[i] == test_data_linear[i]
	
	test_data_array[0] = 100
	expected = [100, 3, 6, 7, 9]
	for i in range(len(test_data_array)):
		assert test_data_array[i] == expected[i]

def test_classifmetric():

	classif_metric = ClassificationMetric(test_data_linear2, test_data_linear)
	assert classif_metric[0].all() == metrics.confusion_matrix(test_data_linear, test_data_linear2).all()
	assert classif_metric[1] == metrics.classification_report(test_data_linear, test_data_linear2)

def test_correlationtest():

	assert all(np.isclose(list(CorrelationTest.anova_oneway(test_data_linear, test_data_linear2).values()), [0.05825242718446602, 0.8153507906592907], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.pearson(test_data_linear, test_data_linear2).values()),  [0.9153061540753287, 0.02920895440940868], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.spearman(test_data_linear, test_data_linear2).values()), [0.9746794344808964, 0.004818230468198537], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.point_biserial(test_data_linear, test_data_linear2).values()), [0.9153061540753287, 0.02920895440940868], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.kendall(test_data_linear, test_data_linear2).values()), [0.9486832980505137, 0.022977401503206086], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.kendall_weighted(test_data_linear, test_data_linear2).values()), [0.9750538072369643, np.nan], rtol=1e-10, equal_nan=True))

def test_fit():

	assert Fit.CircleFit(x=[0,0,-1,1], y=[1, -1, 0, 0]).LSC() == (0.0, 0.0, 1.0, 0.0)

def test_knn():

	model, metric = KNN.knn_classifier(test_data_2D_pairs, test_labels_2D_pairs, 2)
	assert isinstance(model, sklearn.neighbors.KNeighborsClassifier)
	assert np.array([[0,0], [2,0]]).all() == metric[0].all()
	assert '              precision    recall  f1-score   support\n\n           1       0.00      0.00      0.00       0.0\n           2       0.00      0.00      0.00       2.0\n\n    accuracy                           0.00       2.0\n   macro avg       0.00      0.00      0.00       2.0\nweighted avg       0.00      0.00      0.00       2.0\n' == metric[1]
	model, metric = KNN.knn_regressor(test_data_2D_pairs, test_output, 2)
	assert isinstance(model, sklearn.neighbors.KNeighborsRegressor)
	assert (-25.0, 6.5, 2.5495097567963922) == metric

def test_naivebayes():

	model, metric = NaiveBayes.gaussian(test_data_2D_pairs, test_labels_2D_pairs)
	assert isinstance(model, sklearn.naive_bayes.GaussianNB)
	assert metric[0].all() == np.array([[0, 0], [2, 0]]).all()

	model, metric = NaiveBayes.multinomial(test_data_2D_positive, test_labels_2D_pairs)
	assert isinstance(model, sklearn.naive_bayes.MultinomialNB)
	assert metric[0].all() == np.array([[0, 0], [2, 0]]).all()

	model, metric = NaiveBayes.bernoulli(test_data_2D_pairs, test_labels_2D_pairs)
	assert isinstance(model, sklearn.naive_bayes.BernoulliNB)
	assert metric[0].all() == np.array([[0, 0], [2, 0]]).all()

	model, metric = NaiveBayes.complement(test_data_2D_positive, test_labels_2D_pairs)
	assert isinstance(model, sklearn.naive_bayes.ComplementNB)
	assert metric[0].all() == np.array([[0, 0], [2, 0]]).all()

def test_randomforest():

	model, metric = RandomForest.random_forest_classifier(test_data_2D_pairs, test_labels_2D_pairs, 0.3, 2)
	assert isinstance(model, sklearn.ensemble.RandomForestClassifier)
	assert metric[0].all() == np.array([[0, 0], [2, 0]]).all()
	model, metric = RandomForest.random_forest_regressor(test_data_2D_pairs, test_labels_2D_pairs, 0.3, 2)
	assert isinstance(model, sklearn.ensemble.RandomForestRegressor)
	assert metric == (0.0, 1.0, 1.0)

def test_regressionmetric():

	assert RegressionMetric(test_data_linear, test_data_linear2)== (0.7705314009661837, 3.8, 1.9493588689617927)

def test_sort():
	sorts = [Sort.quicksort, Sort.mergesort, Sort.heapsort, Sort.introsort, Sort.insertionsort, Sort.timsort, Sort.selectionsort, Sort.shellsort, Sort.bubblesort, Sort.cyclesort, Sort.cocktailsort]
	for sort in sorts:
		assert all(a == b for a, b in zip(sort(test_data_scrambled), test_data_sorted))

def test_statisticaltest():

	assert StatisticalTest.tukey_multicomparison([test_data_linear, test_data_linear2, test_data_linear3]) == \
		{'group 1 and group 2': [0.32571517201527916, False], 'group 1 and group 3': [0.977145516045838, False], 'group 2 and group 3': [0.6514303440305589, False]}

def test_svm():

	data = test_data_2D_pairs
	labels = test_labels_2D_pairs
	test_data = validation_data_2D_pairs
	test_labels = validation_labels_2D_pairs

	lin_kernel = SVM.PrebuiltKernel.Linear()
	#ply_kernel = SVM.PrebuiltKernel.Polynomial(3, 0)
	rbf_kernel = SVM.PrebuiltKernel.RBF('scale')
	sig_kernel = SVM.PrebuiltKernel.Sigmoid(0)

	lin_kernel = SVM.fit(lin_kernel, data, labels)
	#ply_kernel = SVM.fit(ply_kernel, data, labels)
	rbf_kernel = SVM.fit(rbf_kernel, data, labels)
	sig_kernel = SVM.fit(sig_kernel, data, labels)

	for i in range(len(test_data)):

		assert lin_kernel.predict([test_data[i]]).tolist() == [test_labels[i]]

	#for i in range(len(test_data)):

	#	assert ply_kernel.predict([test_data[i]]).tolist() == [test_labels[i]]

	for i in range(len(test_data)):

		assert rbf_kernel.predict([test_data[i]]).tolist() == [test_labels[i]]

	for i in range(len(test_data)):

		assert sig_kernel.predict([test_data[i]]).tolist() == [test_labels[i]]

def test_equation():

	parser = BNF()
	correctParse = {
		"9": 9.0,
		"-9": -9.0,
		"--9": 9.0,
		"-E": -2.718281828459045,
		"9 + 3 + 6": 18.0,
		"9 + 3 / 11": 9.272727272727273,
		"(9 + 3)": 12.0,
		"(9+3) / 11": 1.0909090909090908,
		"9 - 12 - 6": -9.0,
		"9 - (12 - 6)": 3.0,
		"2*3.14159": 6.28318,
		"3.1415926535*3.1415926535 / 10": 0.9869604400525172,
		"PI * PI / 10": 0.9869604401089358,
		"PI*PI/10": 0.9869604401089358,
		"PI^2": 9.869604401089358,
		"round(PI^2)": 10,
		"6.02E23 * 8.048": 4.844896e+24,
		"e / 3": 0.9060939428196817,
		"sin(PI/2)": 1.0,
		"10+sin(PI/4)^2": 10.5,
		"trunc(E)": 2,
		"trunc(-E)": -2,
		"round(E)": 3,
		"round(-E)": -3,
		"E^PI": 23.140692632779263,
		"exp(0)": 1.0,
		"exp(1)": 2.718281828459045,
		"2^3^2": 512.0,
		"(2^3)^2": 64.0,
		"2^3+2": 10.0,
		"2^3+5": 13.0,
		"2^9": 512.0,
		"sgn(-2)": -1,
		"sgn(0)": 0,
		"sgn(0.1)": 1,
		"sgn(cos(PI/4))": 1,
		"sgn(cos(PI/2))": 0,
		"sgn(cos(PI*3/4))": -1,
		"+(sgn(cos(PI/4)))": 1,
		"-(sgn(cos(PI/4)))": -1,
	}
	for key in list(correctParse.keys()):
		assert parser.eval(key) == correctParse[key]

def test_clustering():

	normalizer = sklearn.preprocessing.Normalizer()

	data = X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

	assert Clustering.dbscan(data, eps=3, min_samples=2).tolist() == [0, 0, 0, 1, 1, -1]
	assert Clustering.dbscan(data, normalizer=normalizer, eps=3, min_samples=2).tolist() == [0, 0, 0, 0, 0, 0]

	data = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])

	assert Clustering.spectral(data, n_clusters=2, assign_labels='discretize', random_state=0).tolist() == [1, 1, 1, 0, 0, 0]
	assert Clustering.spectral(data, normalizer=normalizer, n_clusters=2, assign_labels='discretize', random_state=0).tolist() == [0, 1, 1, 0, 0, 0]