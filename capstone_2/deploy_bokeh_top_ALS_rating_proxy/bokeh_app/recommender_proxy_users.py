#!/usr/bin/env python
# coding: utf-8

# ## Proxy users recommender model

# In[1]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from numpy.random import shuffle
import bz2
import pickle

# Recommender model 1: Proxy User Search
class RecommenderProxyUsers():
    """recommender engine as an estimator"""

    # ******************************************************************
    def __init__(self, n_proxy_users=10):
        """
        Called when initializing the model
        """
        # model parameters
        self.n_proxy_users = n_proxy_users
        
        self.user_data = None
        self.item_data = None
        self.n_top_items = 10
        self.user_factors = None 
        self.item_factors = None 
        self.user_top_rated = None 
        self.item_id_to_index_dict = None
        self.user_info = None

    # ******************************************************************
    def set_params(self, **params):
        self.__dict__.update(params)
        
    # ******************************************************************
    def read_model_data(self, filepath):

        with bz2.BZ2File(filepath, 'rb') as pickle_in:
            [self.user_data, self.item_data] = pickle.load(pickle_in)
            
        # set any games with no categories or mechanics to 'none'
        self.item_data.loc[self.item_data['categories'].isnull(), 'categories'] = 'none'
        self.item_data.loc[self.item_data['mechanics'].isnull(), 'mechanics'] = 'none'

        # column labels for data subsets
        user_top_cols = [col for col in self.user_data.columns if 'top_' in col]
        user_factor_cols = [col for col in self.user_data.columns if 'factor_' in col]
        item_factor_cols = [col for col in self.item_data.columns if 'factor_' in col]
        item_info_cols = [col for col in self.item_data.columns if 'factor_' not in col]

        # split up data for faster processing
        self.user_factors = self.user_data[user_factor_cols].values
        self.user_top_rated = self.user_data[user_top_cols].values 
        
        # number of top items (games) in this dataset
        self.n_top_items = self.user_top_rated.shape[1]
        
        self.item_factors = self.item_data[item_factor_cols].values 
        self.item_info = self.item_data[item_info_cols]
        
        self.item_id_to_index_dict = {key: value for (key, value) in 
                                      zip(self.item_data['id'], 
                                          range(len(self.item_data['id'])))}
        
        self.item_title_to_id_dict = {key: value for (key, value) in 
                                      zip(self.item_data['name'].str.lower(), 
                                          self.item_data['id'].astype(int))}
        
    # ******************************************************************
    def get_tags_from_csv_list(self, taglist):
        """Create df with all unique tags contained in 
            list of csv strings containing multiple tags each.
            Returns tags and counts sorted by most frequent to least."""
        all_tags = []
        for tagset in taglist:
            all_tags += tagset.split(',')
        unique_tags, counts = np.unique(all_tags, return_counts=True)
        return pd.DataFrame( {'tag':unique_tags, 'count':counts} ).sort_values(
            by='count', ascending=False)

    # ******************************************************************
    def get_categories_and_mechanics(self):
        """return lists of all category and mechanic labels"""
        
        # get all categories, sorted by counts
        categories = self.get_tags_from_csv_list(self.item_data['categories'].values)
        
        # remove expansion tag from list
        categories = categories[categories['tag'] != 'Expansion for Base-game']

        # get list of all mechanics, sorted by counts
        mechanics = self.get_tags_from_csv_list(self.item_data['mechanics'].values)
        
        return categories, mechanics
        
    # ******************************************************************
    def get_item_title_id(self, titles):
        """return list of integer item IDs given title names (case insensitive)"""
        return [self.item_title_to_id_dict[title.lower()] for title in titles]
    
    # ******************************************************************
    def get_item_id_index(self, ids):
        """return list of array indices given item IDs"""
        return [self.item_id_to_index_dict[itemid] for itemid in ids]

    # ******************************************************************
    def get_filtered_item_index(self, items, 
                                 weightrange=[1,5],
                                 minrating=1,
                                 categories_include=[],
                                 categories_exclude=[],
                                 mechanics_include=[],
                                 mechanics_exclude=[]):

        # start with all data
        filt_items = items

#         print('filter_data, all data:',filt_items.shape)

        # filter by game weight
        # only filter if not defaults: [1,5]
        if weightrange[0] > 1 or weightrange[1] < 5:
            filt_items = filt_items[ (filt_items['weight'] >= weightrange[0]) &
                             (filt_items['weight'] <= weightrange[1])]
#             print('weightrange, filt_items:',filt_items.shape)

        # filter by lowest average game rating
        # only filter if not default: 1
        if minrating > 1:
            filt_items = filt_items[ filt_items['mean_rating'] >= minrating ]
#             print('minrating, filt_items:',filt_items.shape)

        def tags_in_col(col, taglist):
            return col.apply(lambda x: any(tag in x for tag in taglist))

        # filter by categories to include
        # only filter if not default: [], or ['Any category',...]
        if (len(categories_include) and 
            'Any category' not in categories_include):
            filt_items = filt_items[ tags_in_col(filt_items['categories'], categories_include)]
#             print('categories_include, filt_items:',filt_items.shape)

        # filter by categories to exclude
        # only filter if not default: []
        if len(categories_exclude):
            filt_items = filt_items[ ~(tags_in_col(filt_items['categories'], categories_exclude))]
#             print('categories_exclude, filt_items:',filt_items.shape)

        # filter by mechanics to include
        # only filter if not default: [], or ['Any category',...]
        if (len(mechanics_include) and 
            'Any mechanism' not in mechanics_include):
            filt_items = filt_items[ tags_in_col(filt_items['mechanics'], mechanics_include)]
#             print('mechanics_include, filt_items:',filt_items.shape)

#         print('   filt_items:',filt_items.shape)

        return self.get_item_id_index(filt_items['id'])

    # ******************************************************************
    def get_sorted_proxy_index(self, user_liked):
        liked_idx_set = set(self.get_item_id_index(user_liked))
        scores = [-len(liked_idx_set.intersection(row)) for row in self.user_top_rated]
        return np.argsort(scores)

    # ******************************************************************
    def ratings_from_factors(self, row_index):
        return (np.dot(self.user_factors[row_index,:], self.item_factors.T))
    
    # ******************************************************************
    def recommend_items_by_pref_list(self, liked_item_ids, num2rec=10, **filtargs): 
        
        """Recommend games using multiple liked games in a list of titles.
           This method creates a set of recommended games for each title in prefs and
             then selects the most commonly recommended"""
        
        # get indices to proxy users
        proxy_idx = self.get_sorted_proxy_index(liked_item_ids)

        # average ratings for all items among proxy users
        ratings = np.mean(self.ratings_from_factors(proxy_idx[:self.n_proxy_users]), axis=0)
        
        # Create some randomness here by adding a +/- random 
        #   value to the ratings
        randrange = .2
        randvals = np.random.random(len(ratings))*randrange
        fuzzed_ratings = np.multiply(ratings, randvals)
        
        # get indices of filter allowed items
        filt_item_idx = self.get_filtered_item_index(self.item_info, **filtargs)
        
        def filter_items(item_idx, filter_idx, liked_item_ids):
            """return ordered list of item indices that intersect with filter_idx.
            Also, exclude games in the liked item list"""
            filt_ids = [i for i in item_idx if i in set(filter_idx)]
            return [i for i in filt_ids if not i in set(liked_item_ids)]     
        
        # filtered descending sort of item ratings
        item_idx = filter_items(np.argsort(-fuzzed_ratings), 
                                filt_item_idx, 
                                self.get_item_id_index(liked_item_ids))
        
        # select num2rec top rated game IDs        
        return self.item_data['name'].values[item_idx[:num2rec]]
    
  

