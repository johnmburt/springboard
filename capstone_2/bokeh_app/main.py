#!/usr/bin/env python
# coding: utf-8

# ## Boardgame recommender main script
# 
# NOTE: this only runs as a console command: 
# 
# - bokeh serve --show main.py
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

# Using included state data from Bokeh for map
from bokeh.sampledata.us_states import data as states

# load model
recommender = RecommenderGSS(n_neighbors=10)

# load data
# datadir = './data/'

# get board game data
#  note: this data contains feature data used by model,
#     in future, I should probably calculate the features here at runtime
allgames = pd.read_csv(join(dirname(__file__), 'data', 'bgg_game_data.csv'))
# allgames = pd.read_csv(datadir+'bgg_game_data.csv')
genres = tags_from_csv_list(allgames['categories'].values)
mechanics = tags_from_csv_list(allgames['mechanics'].values)

# Create each of the tabs
tab1 = tab_simple.recommender_tab_simple(recommender, allgames, genres, mechanics)
tab2 = tab_advanced.recommender_tab_advanced(recommender, allgames, genres, mechanics)

# Put all the tabs into one application
tabs = Tabs(tabs = [tab1,tab2])

# Put the tabs in the current document for display
curdoc().add_root(tabs)


# In[ ]:




