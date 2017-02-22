class CreditCard:
	
	def __init__(self, limit):
		self._limit = limit

	def get_limit(self):
		return self._limit


x=CreditCard(1000)

y=x.get_limit()
print(y)