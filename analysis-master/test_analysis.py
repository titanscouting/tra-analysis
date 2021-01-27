import numpy as np
import sklearn
from sklearn import metrics

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
	test_data_2D_positive = np.array([[23, 51], [21, 32], [15, 25], [17, 31]])
	test_output = np.array([1, 3, 4, 5])
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

	classif_metric = ClassificationMetric(test_data_linear2, test_data_linear)
	assert classif_metric[0].all() == metrics.confusion_matrix(test_data_linear, test_data_linear2).all()
	assert classif_metric[1] == metrics.classification_report(test_data_linear, test_data_linear2)

	assert all(np.isclose(list(CorrelationTest.anova_oneway(test_data_linear, test_data_linear2).values()), [0.05825242718446602, 0.8153507906592907], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.pearson(test_data_linear, test_data_linear2).values()),  [0.9153061540753287, 0.02920895440940868], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.spearman(test_data_linear, test_data_linear2).values()), [0.9746794344808964, 0.004818230468198537], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.point_biserial(test_data_linear, test_data_linear2).values()), [0.9153061540753287, 0.02920895440940868], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.kendall(test_data_linear, test_data_linear2).values()), [0.9486832980505137, 0.022977401503206086], rtol=1e-10))
	assert all(np.isclose(list(CorrelationTest.kendall_weighted(test_data_linear, test_data_linear2).values()), [0.9750538072369643, np.nan], rtol=1e-10, equal_nan=True))

	assert Fit.CircleFit(x=[0,0,-1,1], y=[1, -1, 0, 0]).LSC() == (0.0, 0.0, 1.0, 0.0)

	model, metric = KNN.knn_classifier(test_data_2D_pairs, test_labels_2D_pairs, 2)
	assert isinstance(model, sklearn.neighbors.KNeighborsClassifier)
	assert np.array([[0,0], [2,0]]).all() == metric[0].all()
	assert '              precision    recall  f1-score   support\n\n           1       0.00      0.00      0.00       0.0\n           2       0.00      0.00      0.00       2.0\n\n    accuracy                           0.00       2.0\n   macro avg       0.00      0.00      0.00       2.0\nweighted avg       0.00      0.00      0.00       2.0\n' == metric[1]
	model, metric = KNN.knn_regressor(test_data_2D_pairs, test_output, 2)
	assert isinstance(model, sklearn.neighbors.KNeighborsRegressor)
	assert (-25.0, 6.5, 2.5495097567963922) == metric

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

	model, metric = RandomForest.random_forest_classifier(test_data_2D_pairs, test_labels_2D_pairs, 0.3, 2)
	assert isinstance(model, sklearn.ensemble.RandomForestClassifier)
	assert metric[0].all() == np.array([[0, 0], [2, 0]]).all()
	model, metric = RandomForest.random_forest_regressor(test_data_2D_pairs, test_labels_2D_pairs, 0.3, 2)
	assert isinstance(model, sklearn.ensemble.RandomForestRegressor)
	assert metric == (0.0, 1.0, 1.0)

	assert RegressionMetric(test_data_linear, test_data_linear2)== (0.7705314009661837, 3.8, 1.9493588689617927)

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

	svm(test_data_2D_pairs, test_labels_2D_pairs, validation_data_2D_pairs, validation_labels_2D_pairs)
  test_equation()

def svm(data, labels, test_data, test_labels):

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

    test_equation()

def test_equation():

	parser = BNF()

	assert parser.eval("9") == 9.0
	assert parser.eval("-9") == -9.0
	assert parser.eval("--9") == 9.0
	assert parser.eval("-E") == -2.718281828459045
	assert parser.eval("9 + 3 + 6") == 18.0
	assert parser.eval("9 + 3 / 11") == 9.272727272727273
	assert parser.eval("(9 + 3)") == 12.0
	assert parser.eval("(9+3) / 11") == 1.0909090909090908
	assert parser.eval("9 - 12 - 6") == -9.0
	assert parser.eval("9 - (12 - 6)") == 3.0
	assert parser.eval("2*3.14159") == 6.28318
	assert parser.eval("3.1415926535*3.1415926535 / 10") == 0.9869604400525172
	assert parser.eval("PI * PI / 10") == 0.9869604401089358
	assert parser.eval("PI*PI/10") == 0.9869604401089358
	assert parser.eval("PI^2") == 9.869604401089358
	assert parser.eval("round(PI^2)") == 10
	assert parser.eval("6.02E23 * 8.048") == 4.844896e+24
	assert parser.eval("e / 3") == 0.9060939428196817
	assert parser.eval("sin(PI/2)") == 1.0
	assert parser.eval("10+sin(PI/4)^2") == 10.5
	assert parser.eval("trunc(E)") == 2
	assert parser.eval("trunc(-E)") == -2
	assert parser.eval("round(E)") == 3
	assert parser.eval("round(-E)") == -3
	assert parser.eval("E^PI") == 23.140692632779263
	assert parser.eval("exp(0)") == 1.0
	assert parser.eval("exp(1)") == 2.718281828459045
	assert parser.eval("2^3^2") == 512.0
	assert parser.eval("(2^3)^2") == 64.0
	assert parser.eval("2^3+2") == 10.0
	assert parser.eval("2^3+5") == 13.0
	assert parser.eval("2^9") == 512.0
	assert parser.eval("sgn(-2)") == -1
	assert parser.eval("sgn(0)") == 0
	assert parser.eval("sgn(0.1)") == 1
	assert parser.eval("sgn(cos(PI/4))") == 1
	assert parser.eval("sgn(cos(PI/2))") == 0
	assert parser.eval("sgn(cos(PI*3/4))") == -1
	assert parser.eval("+(sgn(cos(PI/4)))") == 1
	assert parser.eval("-(sgn(cos(PI/4)))") == -1
