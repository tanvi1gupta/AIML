class Context(object):

	def __init__(self,name):
		self.lifespan = 2
		self.name = name
		self.active = False

	def activate_context(self):
		self.active = True

	def deactivate_context(self):
		self.active = False

		def decrease_lifespan(self):
			self.lifespan -= 1
			if self.lifespan==0:
				self.deactivate_context()

class FirstGreeting(Context):

	def __init__(self):
		self.lifespan = 1
		self.name = 'FirstGreeting'
		self.active = True

class GetRegNo(Context):

	def __init__(self):
		print('Hi')
		self.lifespan = 1
		self.name = 'GetRegNo'
		self.active = True

class IntentComplete(Context):

	def __init__(self):
		self.lifespan = 1
		self.name = 'IntentComplete'
		self.active = True

class SpellConformation(Context):

	def __init__(self,index,CorrectWord,user_input,context):
		self.lifespan = 1
		self.name = 'SpellConformation'
		self.active = True
		self.index = index
		self.correct = CorrectWord
		self.tobecorrected = user_input
		self.contexttobestored = context

class MovieName(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_moviename'
		self.active = True
class MovieLocation(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_location'
		self.active = True
class Date(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_movieDate'
		self.active = True
class Time(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_movieTime'
		self.active = True
class Theatre(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_theatre'
		self.active = True
class SeatType(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_theatreSeats'
		self.active = True
class Genre(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_movieGenre'
		self.active = True
class Actor(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_movieActor'
		self.active = True
class NumberOfPeople(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_movieNumberOfTickets'
		self.active = True
class Language(Context):
	def __init__(self):
		self.lifespan = 1
		self.name = 'MovieBooking_movieLanguage'
		self.active = True