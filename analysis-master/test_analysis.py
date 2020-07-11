from analysis import analysis as an
from analysis import metrics

def test_():
	test_data_linear = [1, 3, 6, 7, 9]
	y_data_ccu = [1, 3, 7, 14, 21]
	y_data_ccd = [1, 5, 7, 8.5, 8.66]
	test_data_scrambled = [63, 66, 15, 48, 67, 70, 92, 76, 85, 21, 35,  5, 92, 91, 37, 50, 35,
		17, 69, 85, 37, 61, 87,  5, 99,  4, 55, 17,  0, 93, 51, 79, 31, 88,
		47, 27, 13, 61, 75, 93, 16, 73, 22, 65, 32,  9, 33, 97, 88, 27, 87,
		29, 30, 25, 48, 67, 82, 17, 38, 43, 92, 36, 44, 90, 70, 19, 88, 27,
		3,  9, 67, 37, 61, 16, 54, 30, 77, 46, 68, 72, 11, 72, 76, 23, 24,
		2, 35, 96, 33, 24, 10, 51, 33,  0, 37, 42,  1, 43, 58, 61]
	test_data_sorted = [ 0,  0,  1,  2,  3,  4,  5,  5,  9,  9, 10, 11, 13, 15, 16, 16, 17,
		17, 17, 19, 21, 22, 23, 24, 24, 25, 27, 27, 27, 29, 30, 30, 31, 32,
		33, 33, 33, 35, 35, 35, 36, 37, 37, 37, 37, 38, 42, 43, 43, 44, 46,
		47, 48, 48, 50, 51, 51, 54, 55, 58, 61, 61, 61, 61, 63, 65, 66, 67,
		67, 67, 68, 69, 70, 70, 72, 72, 73, 75, 76, 76, 77, 79, 82, 85, 85,
		87, 87, 88, 88, 88, 90, 91, 92, 92, 92, 93, 93, 96, 97, 99]
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
	assert all(a == b for a, b in zip(an.Sort().quicksort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().mergesort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().introsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().heapsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().insertionsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().timsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().selectionsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().shellsort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().bubblesort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().cyclesort(test_data_scrambled), test_data_sorted))
	assert all(a == b for a, b in zip(an.Sort().cocktailsort(test_data_scrambled), test_data_sorted))