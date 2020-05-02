import math

class Glicko2:
	_tau = 0.5

	def getRating(self):
		return (self.__rating * 173.7178) + 1500 

	def setRating(self, rating):
		self.__rating = (rating - 1500) / 173.7178

	rating = property(getRating, setRating)

	def getRd(self):
		return self.__rd * 173.7178

	def setRd(self, rd):
		self.__rd = rd / 173.7178

	rd = property(getRd, setRd)
		
	def __init__(self, rating = 1500, rd = 350, vol = 0.06):

		self.setRating(rating)
		self.setRd(rd)
		self.vol = vol
			
	def _preRatingRD(self):

		self.__rd = math.sqrt(math.pow(self.__rd, 2) + math.pow(self.vol, 2))
		
	def update_player(self, rating_list, RD_list, outcome_list):

		rating_list = [(x - 1500) / 173.7178 for x in rating_list]
		RD_list = [x / 173.7178 for x in RD_list]

		v = self._v(rating_list, RD_list)
		self.vol = self._newVol(rating_list, RD_list, outcome_list, v)
		self._preRatingRD()
		
		self.__rd = 1 / math.sqrt((1 / math.pow(self.__rd, 2)) + (1 / v))
		
		tempSum = 0
		for i in range(len(rating_list)):
			tempSum += self._g(RD_list[i]) * \
						(outcome_list[i] - self._E(rating_list[i], RD_list[i]))
		self.__rating += math.pow(self.__rd, 2) * tempSum
		
		
	def _newVol(self, rating_list, RD_list, outcome_list, v):

		i = 0
		delta = self._delta(rating_list, RD_list, outcome_list, v)
		a = math.log(math.pow(self.vol, 2))
		tau = self._tau
		x0 = a
		x1 = 0
		
		while x0 != x1:
			# New iteration, so x(i) becomes x(i-1)
			x0 = x1
			d = math.pow(self.__rating, 2) + v + math.exp(x0)
			h1 = -(x0 - a) / math.pow(tau, 2) - 0.5 * math.exp(x0) \
			/ d + 0.5 * math.exp(x0) * math.pow(delta / d, 2)
			h2 = -1 / math.pow(tau, 2) - 0.5 * math.exp(x0) * \
			(math.pow(self.__rating, 2) + v) \
			/ math.pow(d, 2) + 0.5 * math.pow(delta, 2) * math.exp(x0) \
			* (math.pow(self.__rating, 2) + v - math.exp(x0)) / math.pow(d, 3)
			x1 = x0 - (h1 / h2)

		return math.exp(x1 / 2)
		
	def _delta(self, rating_list, RD_list, outcome_list, v):

		tempSum = 0
		for i in range(len(rating_list)):
			tempSum += self._g(RD_list[i]) * (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
		return v * tempSum
		
	def _v(self, rating_list, RD_list):

		tempSum = 0
		for i in range(len(rating_list)):
			tempE = self._E(rating_list[i], RD_list[i])
			tempSum += math.pow(self._g(RD_list[i]), 2) * tempE * (1 - tempE)
		return 1 / tempSum
		
	def _E(self, p2rating, p2RD):

		return 1 / (1 + math.exp(-1 * self._g(p2RD) * \
									(self.__rating - p2rating)))
		
	def _g(self, RD):

		return 1 / math.sqrt(1 + 3 * math.pow(RD, 2) / math.pow(math.pi, 2))
		
	def did_not_compete(self):

		self._preRatingRD()