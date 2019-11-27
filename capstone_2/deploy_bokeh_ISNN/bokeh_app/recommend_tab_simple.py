#!/usr/bin/env python
# coding: utf-8

# ## Simple recommend tab
# 
# Next steps:
# 
# - style and formatting
# - format recommendations: icon, link to BGG page, link to buy
# - plot games
# - plot games 3D interactive
# 

# In[86]:


# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.models import (CategoricalColorMapper, HoverTool, 
    ColumnDataSource, Panel, 
    FuncTickFormatter, SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, AutocompleteInput, 
      Tabs, CheckboxButtonGroup, Div, Button,
      TableColumn, DataTable, Select)
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.palettes import Category20_16


# In[1]:


def recommender_tab_simple(recommender, allgames, categories, mechanics):

    # create a list of divs
    def make_div_list(textlist, max_lines, fmt_str="""%s""", **attribs):
        """create a list of divs containing text to display"""
        divs = []
        for i in range(max_lines):
            if len(textlist) > i:
                divs.append(Div(text=fmt_str%(textlist[i]), **attribs)) 
            else:
                divs.append(Div(text=fmt_str%(' '), **attribs))
        return divs

    def make_rec_list(titles, max_lines):
        """create a recommendation list of games,
        with a thumbnail, game title, info and Amazon buy links"""
        global games_by_title
        fmt_str1="""
            <div class="rec-post-container">                
                <div class="rec-post-thumb"><img src="%s" /></div>
                <div class="rec-post-content">
                    <h3 class="rec-post-title">%s<br>
                    <a href="%s" target="_blank">Info</a><span>&nbsp;&nbsp;</span>
                    <a href="%s" target="_blank">Buy on Amazon</a> </h3>
                </div>
            </div>"""        
        fmt_str2=""""""
        divs = []
        for i in range(max_lines):
            # there is a title available for this list slot
            if len(titles) > i:
                divs.append(Div(text=fmt_str1%(
                    games_by_title['pic_url'].loc[titles[i]],
                    titles[i],
                    'https://boardgamegeek.com/boardgame/' + 
                        str(games_by_title['id'].loc[titles[i]]),
                    'https://www.amazon.com/s?k=' + titles[i].replace(' ','+') + 
                        '&i=toys-and-games'
                ))) 
            # there no title available for this list slot
            else:
                divs.append(Div(text=fmt_str2))
        return divs
        
    # update the 'liked games' list UI elements
    def update_liked_list(titlelist):
        global max_liked
        ctl_liked_games.children = make_div_list(titlelist, 
                                                 max_liked, 
                                                 fmt_str=liked_list_fmt, 
                                                 render_as_text=False)
        
    # update the 'recommended games' list UI elements
    def update_recommended_list(titlelist):
        global n_recommendations
        ctl_recommended_games.children = make_rec_list(titlelist, 
                                                 n_recommendations)

    # called when a control widget is changed
    def update_preflist(attr, old, new):
        global liked_games
        liked_games.append(ctl_game_entry.value)
        liked_games = list(filter(None, set(liked_games)))
        # get control values
        update_liked_list(liked_games)
        ctl_game_entry.value = ''

    # clear out the list of preferred games
    def reset_preferred_games():
        global liked_games
        liked_games = []
        update_liked_list(liked_games)        
        
    # user wants some recommendations (clicked the rec button)
    def recommend_games():
        global liked_games, recommended_games
        global games_all, n_recommendations, title_list
        global title_list_lower
        
        # get some default filter parameters:
        weight = []
        minrating = 6.5
        categories = ['Any category']
        mechanics = ['Any mechanism']
        for title in liked_games:
            idx = (np.array(title_list_lower) 
                   == title.lower()).nonzero()[0][0]
            info = games_all.iloc[idx,:]
            weight.append(info['weight'])
            categories += info['categories'].split(',')
            mechanics += info['mechanics'].split(',')

        # select a range of weights around the liked game weights
        weightrange = [max(1,np.min(weight)-0.25),
                       min(5,np.max(weight)+0.25)]
        
        # select games to search from based on filters:
        recommended_games = recommender.recommend_games_by_pref_list(
            liked_games, games_all, num2rec=n_recommendations,
             weightrange=weightrange,
             minrating=minrating,
             categories_include=categories,
             categories_exclude=['Expansion for Base-game'],
             mechanics_include=mechanics,
             mechanics_exclude=[]
            )

        update_recommended_list(recommended_games)   

    # NOTE: I'm using globals because I'm running into variable scope
    #  problems with the bokeh handlers. Easiest to declare globals
    global liked_games, recommended_games, games_all 
    global n_recommendations, max_liked, title_list, title_list_lower
    global games_by_title
    
    # layout params
    n_recommendations = 5
    max_liked = 8
    # Format to use for liked list. 
    # This needs to be changed to work like rec list
    liked_list_fmt = """<div style="font-size : 14pt; line-height:14pt;">%s</div>"""

    # variables used by the tab
    liked_games = []
    recommended_games = []
    weight_range = [1,5]
    games_all = allgames # use all games for search 
    games_by_title = allgames.set_index('name')
        
    # list of all game titles
    title_list = games_all['name']
    title_list_lower = [s.lower() for s in title_list]
    
    # preferred game entry text control
    ctl_game_entry = AutocompleteInput(
        completions=list(title_list)+list(title_list_lower),
        min_characters = 1,                               
        title = 'Enter some game names you like:')
    ctl_game_entry.on_change('value', update_preflist)
    
    # reset liked game list button
    ctl_reset_prefs = Button(label = 'Reset game list',
                             width_policy='min', align='end')
    ctl_reset_prefs.on_click(reset_preferred_games)
    
    # liked list title
    ctl_liked_list_title = Div(text=
        """<div style="font-size : 18pt; line-height:16pt;">Games you like:</div>""")
   
    # liked game entries
    ctl_liked_games = WidgetBox(children=make_div_list(liked_games, max_liked, 
        fmt_str=liked_list_fmt))
    
    # recommended list title
    ctl_recommended_list_title = Div(text=
        """<div style="font-size : 18pt; line-height:16pt;">Games we recommend:</div>""")
    
    # recommended games list widget
    ctl_recommended_games = WidgetBox(children=
        make_rec_list(recommended_games, n_recommendations) )
    
    # Recommend games button
    ctl_recommend = Button(label = 'Recommend some games!',width_policy='min')
    ctl_recommend.on_click(recommend_games)
    
    # controls to select preferred games
    pref_controls = WidgetBox(
        ctl_liked_list_title,
        ctl_liked_games, 
        Spacer(min_height=20),
        ctl_game_entry, 
        ctl_reset_prefs,
        Spacer(min_height=40),
        ctl_recommend)
        
    # recommendation results
    results_controls = WidgetBox(
        ctl_recommended_list_title,
        ctl_recommended_games)
    
    # Create a row layout
    layout = row(pref_controls, results_controls)
    
    # Make a tab with the layout   
    tab = Panel(child=layout, title = 'Simple Game Recommender')
    
    return tab


# In[ ]:




