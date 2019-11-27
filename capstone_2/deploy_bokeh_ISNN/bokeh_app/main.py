#!/usr/bin/env python
# coding: utf-8

# ## Boardgame recommender main script
# 
# NOTE: this only runs as a console command: 
# 
# - bokeh serve --show bokeh_app/
# 
# If run in Jupyter, it will give an error: "name '__file__' is not defined"
# 

# In[1]:


# Pandas for data management
import pandas as pd
import numpy as np

# os methods for manipulating paths
from os.path import dirname, join

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs

# Each tab is drawn by one script
# import local scripts for the tabs.
# I use reload for debugginf to force 
#   reload of changed modules
import recommender
import recommend_tab_simple
import recommend_tab_advanced

import importlib
importlib.reload(recommender)
importlib.reload(recommend_tab_simple)
importlib.reload(recommend_tab_advanced)

from recommender import RecommenderGSS
import recommend_tab_simple as tab_simple
import recommend_tab_advanced as tab_advanced

def tags_from_csv_list(taglist):
    """Create df with all unique tags contained in 
        list of csv strings containing multiple tags each.
        Returns tags and counts sorted by most frequent to least."""
    all_tags = []
    for tagset in taglist:
        all_tags += tagset.split(',')
    unique_tags, counts = np.unique(all_tags, return_counts=True)
    return pd.DataFrame( {'tag':unique_tags, 'count':counts} ).sort_values(
        by='count', ascending=False)

# load data
datadir = './data/'

# get board game data
allgames = pd.read_hdf(join(dirname(__file__), 'data', 
                            'bgg_game_data_big.h5'))

# set any games with no categories or mechanics to 'none'
allgames.loc[allgames['categories'].isnull(), 'categories'] = 'none'
allgames.loc[allgames['mechanics'].isnull(), 'mechanics'] = 'none'

# get all categories, sorted by counts
categories = tags_from_csv_list(allgames['categories'].values)
categories = categories[categories['tag'] != 'Expansion for Base-game']

# get list of all mechanics, sorted by counts
mechanics = tags_from_csv_list(allgames['mechanics'].values)

# Number of neighbors to search when selecting recommendations.
# This number matters less than I'd have thought. Tunes to ~8-10, 
#  but I'm setting it to 50 here to generate some randomness 
#  in the recs: the alg randomly selects n_recs from nearest n_neighbors
n_neighbors = 50

# number of game space features in this dataset
# NOTE: it may be good to limit search dims < # features
n_search_dims = len([s for s in allgames.columns if 'f_' in s])

# load model
recommender = RecommenderGSS(n_neighbors=n_neighbors, n_search_dims=n_search_dims)

# Create each of the tabs
tab1 = tab_simple.recommender_tab_simple(recommender, allgames, categories, mechanics)
tab2 = tab_advanced.recommender_tab_advanced(recommender, allgames, categories, mechanics)

# Put all the tabs into one application
tabs = Tabs(tabs = [tab1,tab2])
# tabs = Tabs(tabs = [tab2,tab1])

# Put the tabs in the current document for display
curdoc().add_root(tabs)

