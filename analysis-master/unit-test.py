from analysis import analysis as an
test_data_linear = [1, 3, 6, 7, 9]
y_data_ccu = [1, 3, 7, 14, 21]
y_data_ccd = [1, 5, 7, 8.5, 8.66]
assert an.basic_stats(test_data_linear) == {"mean": 5.2, "median": 6.0, "standard-deviation": 2.85657137141714, "variance": 8.16, "minimum": 1.0, "maximum": 9.0}
assert an.z_score(3.2, 6, 1.5) == -1.8666666666666665
assert an.z_normalize(test_data_linear).tolist() == [5.2, 6.0, 2.85657137141714, 8.16, 1.0, 9.0]
assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["lin"])) == True
assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccd, ["log"])) == True
assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["exp"])) == True
assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccu, ["ply"])) == True
assert all(isinstance(item, str) for item in an.regression(test_data_linear, y_data_ccd, ["sig"])) == True
assert an.Metric().elo(1500, 1500, [1, 0], 400, 24) == 1512.0
assert an.Metric().glicko2(1500, 250, 0.06, [1500, 1400], [250, 240], [1, 0]) == (1478.864307445517, 195.99122679202452, 0.05999602937563585)
assert an.Metric().trueskill([[(25, 8.33), (24, 8.25), (32, 7.5)], [(25, 8.33), (25, 8.33), (21, 6.5)]], [1, 0]) == [(an.metrics.trueskill.Rating(mu=21.346, sigma=7.875), an.metrics.trueskill.Rating(mu=20.415, sigma=7.808), an.metrics.trueskill.Rating(mu=29.037, sigma=7.170)), (an.metrics.trueskill.Rating(mu=28.654, sigma=7.875), an.metrics.trueskill.Rating(mu=28.654, sigma=7.875), an.metrics.trueskill.Rating(mu=23.225, sigma=6.287))]