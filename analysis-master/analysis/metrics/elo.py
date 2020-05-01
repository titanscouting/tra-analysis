import numpy as np

def calculate(starting_score, opposing_score, observed, N, K):

		expected = 1/(1+10**((np.array(opposing_score) - starting_score)/N))

		return starting_score + K*(np.sum(observed) - np.sum(expected))