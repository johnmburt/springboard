#!/usr/bin/env python
# coding: utf-8

# ## Recommender models

# In[15]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


# Recommender model 1: Game Space Search
class RecommenderGSS():
    """recommender engine as an estimator"""

    # ******************************************************************
    def __init__(self, n_neighbors=10):
        """
        Called when initializing the model
        """
        # model parameters
        self.n_neighbors = n_neighbors  # number of neighbor titles to search for

    # ******************************************************************
    def set_params(self, **params):
        self.__dict__.update(params)

    # ******************************************************************
    def find_nearest_neighbors(self, coords, x, numnearest):
        """Brute force nearest neighbor search"""

        # get euclidean distances of all points to x
        # dists = cdist(np.reshape(x, (1, -1)), coords)
        dists = cdist(x, coords)

        # sort the distances
        ind, = np.argsort(dists)

        # return the numnearest nearest neighbors
        return ind[:numnearest]

    # ******************************************************************
    def recommend_games_by_one_title(self, targettitle, game_data, num2rec=1):
        """Recommend games based on nearest neighbor to one game title"""

#         print('recommend_games_by_one_title:',targettitle)
        gametitles = game_data['name'].values
        coords = game_data[[s for s in game_data.columns if 'f_' in s]].values
    
        # get coords of target title,
        # use case insensitive search
        targetindex = (np.array([s.lower() for s in gametitles]) == targettitle.lower()
                      ).nonzero()[0]
#         print(targetindex, gametitles[targetindex])
        targetcoord = coords[targetindex, :]
#         print('targetcoord', targetcoord)

        if targetcoord.shape[0] == 0:
            return []
        
        # find nearest neighbors
#         print(coords.shape, targetcoord.shape)
        ind = self.find_nearest_neighbors(coords, targetcoord, 
                                          max(self.n_neighbors, num2rec+1))
        # ind = self.find_nearest_neighbors(coords, targetcoord, num2rec + 1)
#         print('ind.shape',ind.shape)
#         print(ind[1:num2rec+1])
        # Note: first entry will be the target title (distance 0)
#         print('returned: ', list(ind[1:num2rec+1]))
        return list(ind[1:num2rec+1])

    # ******************************************************************
    def recommend_games_by_prefs_sets(self, pref, game_data, num2rec=10):
        """Recommend games using multiple liked and disliked games.
           This method creates a set of recommended games for each title in prefs and
             then selects the most commonly recommended,
             excluding any recs based on disliked games"""

        recs = []
        for title in pref['like']:
            recs.extend(self.recommend_games_by_one_title(title, self.n_neighbors))
        unique, counts = np.unique(recs, return_counts=True)
        recs = (np.array([unique, counts])[0, np.argsort(-counts)].T)

        norecs = []
        for title in pref['dislike']:
            norecs.extend(self.recommend_games_by_one_title(title, game_data, self.n_neighbors))
        norecs = list(np.unique(norecs))

        allrecs = []
        for r in recs:
            if ~any(r == norecs):
                allrecs.append(r)

        return allrecs[:num2rec]
    
    # ******************************************************************
    def remove_prefs_from_recs(self, preflist, game_data, recs):
        pass

    # ******************************************************************
    def recommend_games_by_pref_list(self, preflist, game_data, num2rec=10):
        """Recommend games using multiple liked games in a list of titles.
           This method creates a set of recommended games for each title in prefs and
             then selects the most commonly recommended"""

        recs = []
        for title in preflist:
            recs.extend(self.recommend_games_by_one_title(title, game_data, self.n_neighbors))
#         print('recs',recs)
        # NOTE: np.unique sorts the results, which could create a bias for older games 
        unique, counts = np.unique(recs, return_counts=True)
#         print('unique, counts', unique, counts)
        recs = (np.array([unique, counts])[0, np.argsort(-counts)].T)
#         print('recs',recs)
        return recs[:num2rec]
    
 


# In[ ]:




