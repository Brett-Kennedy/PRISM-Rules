import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs, fetch_openml
import sklearn.datasets as datasets

import sys
sys.path.append('../')
from prism_rules import PrismRules


def test_prism():
	# vals = [0, 1, 2, 3, 4, 5]
	# a_arr = np.random.choice(vals, 1000)
	# b_arr = np.random.choice(vals, 1000)
	# c_arr = np.random.choice(vals, 1000)
	# y = [True if ((a >= 2) and (b >= 4) and (c <= 2)) else False for a, b, c in zip(a_arr, b_arr, c_arr)]
	# df = pd.DataFrame({"a": a_arr, "b": b_arr, "c": c_arr, "Y": y})
	# prism = PrismRules()
	# prism.get_prism_rules(df, 'Y', display_stats=True)

	bunch = make_classification(n_samples=1000, flip_y=0, random_state=0)
	df = pd.DataFrame(bunch[0])
	df['Y'] = bunch[1]
	prism = PrismRules(min_coverage=100)
	_ = prism.get_prism_rules(df, 'Y')
	prism.get_bin_ranges()

	# df = pd.DataFrame(
	# 	{
	# 		"age":    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
	# 		"specRX": [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
	# 		"astig":  [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
	# 		"tears":  [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
	# 		"class":  [3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 3, 3, 1, 3, 2, 3, 3]
	# 	}
	# )
	# rules = get_prism_rules(df, 'class')

	# data = datasets.load_diabetes()
	# df = pd.DataFrame(data.data, columns=data.feature_names)
	# df['Y'] = data['target']

	# data = fetch_openml('ipums_la_98-small')
	# df = pd.DataFrame(data.data, columns=data.feature_names)
	# df['Y'] = data.target
	# prism = PrismRules(min_coverage=100, verbose=1)
	# _ = prism.get_prism_rules(df, 'Y')

	# bunch = make_classification(
	# 	n_samples=10_000,
	# 	n_features=20,
	# 	n_informative=1,
	# 	n_redundant=0,
	# 	n_repeated=0,
	# 	n_classes=3,
	# 	n_clusters_per_class=2,
	# 	class_sep=2,
	# 	flip_y=0,
	# 	random_state=0
	# )
	# df = pd.DataFrame(bunch[0])
	# df['Y'] = bunch[1]
	# prism = PrismRules(min_coverage=500, min_prob=0.9, verbose=1)
	# _ = prism.get_prism_rules(df, 'Y')


if __name__ == '__main__':
	test_prism()