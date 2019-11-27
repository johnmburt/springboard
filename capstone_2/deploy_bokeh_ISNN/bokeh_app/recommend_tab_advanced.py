#!/usr/bin/env python
# coding: utf-8

# ## Advanced recommend tab
# 
# 
# Next steps:
# 
# - style and formatting
# - format recommendations: icon, link to BGG page, link to buy
# - plot games
# - plot games 3D interactive
# 

# In[2]:


# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.models import (CategoricalColorMapper, HoverTool, 
    ColumnDataSource, Panel, 
    FuncTickFormatter, SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, AutocompleteInput, 
      Tabs, CheckboxButtonGroup, Div, Button, MultiSelect, 
      TableColumn, DataTable, Select, RangeSlider, Slider)
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.palettes import Category20_16

def recommender_tab_advanced(recommender, allgames, categories, mechanics):

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
            # no title, so fill with blank
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
    def update_filters(attr, old, new):
        global category_includes, mechanics_includes
        global category_excludes, mechanics_excludes

        category_includes = [
            ctl_category_selection1.labels[i] for i in ctl_category_selection1.active]
        category_includes += [
            ctl_category_selection2.labels[i] for i in ctl_category_selection2.active]

        mechanics_includes = [
            ctl_mechanics_selection1.labels[i] for i in ctl_mechanics_selection1.active]
        mechanics_includes += [
            ctl_mechanics_selection2.labels[i] for i in ctl_mechanics_selection2.active]

        # NOTE: this will need to be changed if I ever implement exclude selections!
        if ctl_include_expansions.active:
            category_excludes = []
        else:
            category_excludes = ['Expansion for Base-game']           
        
    # called when a control widget is changed
    def update_preflist(attr, old, new):
        global liked_games
        liked_games.append(ctl_game_entry.value)
        liked_games = list(filter(None, set(liked_games)))
        # get control values
        update_liked_list(liked_games)
        ctl_game_entry.value = ''

    # reset preferred games list
    def reset_preferred_games():
        global liked_games
        liked_games = []
        update_liked_list(liked_games)        
        
    # recommend some games
    def recommend_games():
        global liked_games, recommended_games
        global games_all, n_recommendations, title_list
        global category_includes, mechanics_includes
        
        # select games to search from based on filters:
        recommended_games = recommender.recommend_games_by_pref_list(
            liked_games, games_all, num2rec=n_recommendations,
             weightrange=ctl_game_weight.value,
             minrating=ctl_game_min_rating.value,
             categories_include=category_includes,
             categories_exclude=category_excludes,
             mechanics_include=mechanics_includes,
             mechanics_exclude=mechanics_excludes
            )
        
        # show the recommended games
        update_recommended_list(recommended_games)   

    # NOTE: I'm using globals because I'm running into variable scope
    #  problems with the bokeh handlers. Easiest to declare globals
    global liked_games, recommended_games, games_all
    global n_recommendations, max_liked, title_list, title_list_lower
    global category_includes, mechanics_includes
    global category_excludes, mechanics_excludes
    global games_by_title
    
    # layout params
    n_recommendations = 5
    max_liked = 8
    num_check_options = 20
    
    # Format to use for liked list. 
    # This needs to be changed to work like rec list
    liked_list_fmt = """<div style="font-size : 14pt; line-height:14pt;">%s</div>"""
    
    # variables used by the tab
    games_all = allgames # use all games for search     
    liked_games = []
    recommended_games = []
    weight_range = [1,5]
    category_includes = []
    mechanics_includes = []
    category_excludes = []
    mechanics_excludes = []

    # list of all game titles
    title_list = games_all['name']
    title_list_lower = [s.lower() for s in title_list]
    games_by_title = allgames.set_index('name')
    
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
    ctl_recommend = Button(label = 'Recommend some games!',
                           width_policy='min', align='center')
    ctl_recommend.on_click(recommend_games)
    
    # game weight slider
    ctl_game_weight = RangeSlider(start = 1, end = 5, value = (1, 5),
        step = .1, title = 'Game weight range',width_policy='min',)
    ctl_game_weight.on_change('value', update_filters)
    
    # min game rating slider
    ctl_game_min_rating = Slider(start = 1, end = 10, value = 7,
        step = .1, title = 'Minimum average rating', 
                                 width_policy='min')
    ctl_game_min_rating.on_change('value', update_filters)
    
    # game category selection
    category_list = ['Any category'] + list(categories['tag'].values)    
    ctl_category_selection1 = CheckboxGroup(
        labels=category_list[:int(num_check_options/2)], 
        width_policy='min', active = [0])
    ctl_category_selection1.on_change('active', update_filters)
    ctl_category_selection2 = CheckboxGroup(
        labels=category_list[int(num_check_options/2):num_check_options], 
        width_policy='min')
    ctl_category_selection2.on_change('active', update_filters)

    # game mechanism checkbox group
    mechanics_list = ['Any mechanism'] + list(mechanics['tag'].values)
    ctl_mechanics_selection1 = CheckboxGroup(
        labels=mechanics_list[:int(num_check_options/2)], 
        width_policy='min', active = [0])
    ctl_mechanics_selection1.on_change('active', update_filters)
    ctl_mechanics_selection2 = CheckboxGroup(
        labels=mechanics_list[int(num_check_options/2):num_check_options], 
        width_policy='min')
    ctl_mechanics_selection2.on_change('active', update_filters)
        
    # select whether to include expansions
    ctl_include_expansions = CheckboxGroup(labels=['Include game expansions'],
                                           width_policy='min')
    
    ctl_include_expansions.on_change('active', update_filters)
    # controls to select preferred games
    pref_controls = WidgetBox(
        ctl_liked_list_title,
        ctl_liked_games, 
        Spacer(min_height=20),
        ctl_game_entry, 
        ctl_reset_prefs,
        Spacer(min_height=5),
        ) 
    
    ctl_liked_list_title = Div(text=
        """<div style="font-size : 18pt; line-height:16pt;">Game Categories:</div>""")

    filter_controls = WidgetBox(
        row(ctl_game_weight, Spacer(min_width=50), ctl_game_min_rating),
        row(ctl_include_expansions),
        column(
            row(Div(text="""<div style="font-size : 18pt; line-height:16pt;">Game Categories:</div>"""),
                Spacer(min_width=50), ctl_recommend),
            row(ctl_category_selection1, ctl_category_selection2),
            Spacer(min_height=5),
            Div(text="""<div style="font-size : 18pt; line-height:16pt;">Game Mechanics:</div>"""),
            row(ctl_mechanics_selection1, ctl_mechanics_selection2),
            )
        )
    
    # recommendation results
    results_controls = WidgetBox(
        ctl_recommended_list_title,
        ctl_recommended_games,
        Spacer(min_height=10),
        )
    
    # Create a row layout
    layout = row(column(pref_controls,filter_controls), 
                 Spacer(min_width=50), results_controls)
    
    # Make a tab with the layout   
    tab = Panel(child=layout, title = 'Advanced Game Recommender')
    
    return tab


# In[ ]:




