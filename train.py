import sys
sys.path.insert(0, '/Users/SUMOTEKHNOLOGISOLUSI/Projects/personal/python/flurs')

import pickle
from os import listdir
from collections import deque

import numpy as np
from flurs.data.entity import User, Item, Event
from flurs.recommender.mf import MFRecommender

recommender = MFRecommender(k=2)
recommender.init_recommender()
item_buffer = deque()

def batch_evaluate(test_events, repeat=False):
    """Evaluate the current model by using the given test events.
    Args:
        test_events (list of Event): Current model is evaluated by these events.
    Returns:
        float: Mean Percentile Rank for the test set.
    """
    percentiles = np.zeros(len(test_events))
    all_items = set(item_buffer)
    for i, e in enumerate(test_events):
        unobserved = all_items
        if not repeat:
            # make recommendation for all unobserved items
            unobserved -= recommender.users[e.user.index]['known_items']

            # true item itself must be in the recommendation candidates
            unobserved.add(e.item.index)

        candidates = np.asarray(list(unobserved))
        recos, scores = recommender.recommend(e.user, candidates)

        pos = np.where(recos == e.item.index)[0][0]
        percentiles[i] = pos / (len(recos) - 1) * 100

    return np.mean(percentiles)

def batch_update(train_events, test_events, n_epoch):
    """Batch update called by the fitting method.
    Args:
        train_events (list of Event): Positive training events.
        test_events (list of Event): Test events.
        n_epoch (int): Number of epochs for the batch training.
    """
    for epoch in range(n_epoch):
        # SGD requires us to shuffle events in each iteration
        # if n_epoch == 1
        #   => shuffle is not required because it is a deterministic training (i.e. matrix sketching)
        if n_epoch != 1:
            np.random.shuffle(train_events)

        # train
        print('#################### training epoch: '+ str(epoch) +' ####################')
        for e in train_events:
            recommender.update_recommender(e, batch_train=True)

        # test
        MPR = batch_evaluate(test_events)
        print('epoch %2d: MPR = %f' % (epoch + 1, MPR))

# prepare events data for batch training
events = []
docs = [f for f in listdir('data') if f.endswith('.txt')]
for _idx, doc in enumerate(docs):
    print('processing...'+ str(_idx) +' '+ doc)
    content = open('data/' + doc, 'r').read().split(',')

    # add user
    user = User(_idx)
    if recommender.is_new_user(_idx):
        recommender.add_user(user)

    for _idt, val in enumerate(content):
        if val == '1':
            item = Item(_idt)

            # add item
            if recommender.is_new_item(_idt):
                recommender.add_item(item)

            # add event
            event = Event(user, item, float(val))
            events.append(event)

# split events data by percentage
percentage = 0.7 # 70%
n = int(round(percentage * len(events)))
train_events = events[:n]
test_events = events[n:]

# make initial status for batch training
for e in train_events:
    recommender.users[e.user.index]['known_items'].add(e.item.index)
    item_buffer.append(e.item.index)

# for batch evaluation, temporarily save new users info
for e in test_events:
    item_buffer.append(e.item.index)

# do the training using n_epoch = 99
batch_update(train_events, test_events, 99)

# batch test events are considered as a new observations;
# the model is incrementally updated based on them before the incremental evaluation step
for e in test_events:
    recommender.users[e.user.index]['known_items'].add(e.item.index)
    recommender.update_recommender(e)

# save object
file_name = 'recommender.pickle'
file_object = open(file_name, 'wb')
pickle.dump(recommender, file_object)
file_object.close()
