import sys
sys.path.insert(0, '/Users/SUMOTEKHNOLOGISOLUSI/Projects/personal/python/flurs')

import pickle
import numpy as np
from flurs.data.entity import User

# load object
file_name = 'recommender.pickle'
file_object = open(file_name, 'rb')
recommender = pickle.load(file_object)

user = User(0)
print(recommender.recommend(user, np.arange(0, 39, 1)))

user = User(1)
print(recommender.recommend(user, np.arange(0, 39, 1)))

user = User(2)
print(recommender.recommend(user, np.arange(0, 39, 1)))
