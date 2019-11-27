#!/usr/bin/env python
# coding: utf-8

# ## Boardgame recommender main script
# 
# NOTE: this only runs as a console command: 
# 
# - bokeh serve bokeh_app --show
# 
# If run in Jupyter, it will throw "NameError: name '__file__' is not defined"
# 

# In[19]:


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
import recommender_proxy_users
import recommend_tab_simple
import recommend_tab_advanced

import importlib
importlib.reload(recommender_proxy_users)
importlib.reload(recommend_tab_simple)
importlib.reload(recommend_tab_advanced)

from recommender_proxy_users import RecommenderProxyUsers
import recommend_tab_simple as tab_simple
import recommend_tab_advanced as tab_advanced

# load data
datadir = './data/'

# Number of proxy users in ALS factored rating data to average
# game ratings for when selecting "best game" recommendations.
# Hyperparam tuning found that about 100 proxy users works well.
n_proxy_users = 100

# relative file path to recommender data
datafilepath = join(dirname(__file__), 'data', 'bgg_pu_data_all.pkl')
# datafilepath = datadir+'bgg_pu_data_all.pkl'

# load model
recommender = RecommenderProxyUsers(n_proxy_users=n_proxy_users)
recommender.read_model_data(datafilepath)

# Create each of the tabs
tab1 = tab_simple.recommender_tab_simple(recommender)
tab2 = tab_advanced.recommender_tab_advanced(recommender)

# Put all the tabs into one application
tabs = Tabs(tabs = [tab1,tab2])
# tabs = Tabs(tabs = [tab2,tab1]) # for debugging advanced tab

# Put the tabs in the current document for display
curdoc().add_root(tabs)

