# Springboard Capstone 2 project: Building a Board Game Recommendation Engine

Notebook descriptions:

|  Notebook  |  Description  |
|  --  |  --  |
| [recsys_data_prep_1_BGG_download_game_ids_vf.ipynb](recsys_data_prep_1_BGG_download_game_ids_vf.ipynb) | Download board game IDs from boardgamegeek.com |
| [recsys_data_prep_2_BGG_download_game_info_vf.ipynb](recsys_data_prep_2_BGG_download_game_info_vf.ipynb) | Download board game metadata from boardgamegeek.com |
| [recsys_data_prep_3_BGG_download_user_ratings_vf.ipynb](recsys_data_prep_3_BGG_download_user_ratings_vf.ipynb) | Download user ratings for selected boardgames from boardgamegeek.com |
| [recsys_data_prep_4_gen_unfilled_utility_mx_vf.ipynb](recsys_data_prep_4_gen_unfilled_utility_mx_vf.ipynb) | Create a matrix of ratings for item (game) x user: the utility matrix |
| [recsys_data_prep_5_gen_ALS_filled_utility_mx_vf.ipynb](recsys_data_prep_5_gen_ALS_filled_utility_mx_vf.ipynb) | Matrix Factorization with Alternating Least Squares |
| [recsys_data_prep_6_gen_game_coords_vf.ipynb](recsys_data_prep_6_gen_game_coords_vf.ipynb) | Create complete game info file for recommender app, including SVD features |
| [recsys_utilities.ipynb](recsys_utilities.ipynb) | Project utility functions |
|  --  |  --  |
| [recsys_method_1_ISNN_tuning_vf.ipynb](recsys_method_1_ISNN_tuning_vf.ipynb) | Recommendation model 1: Item Search by Nearest Neighbors (ISNN) | 
| [recsys_method_2a_top_ALS_rating_foldin_tuning_vf.ipynb](recsys_method_2a_top_ALS_rating_foldin_tuning_vf.ipynb) | Recommendation model 2a, Top ALS Rating: fold-in new user to utility matrix, select highest rated games | 
| [recsys_method_2b_top_ALS_rating_proxy_user_tuning_vf.ipynb](recsys_method_2b_top_ALS_rating_proxy_user_tuning_vf.ipynb) | Recommendation model 2b, Top ALS Rating: use existing users in filled utility matrix as proxies to new user | 
|  --  |  --  |
| [deploy_bokeh_ISNN](./deploy_bokeh_ISNN) | Folder with Bokeh server based board game recommender web app, Item Search Nearest Neighbors method |
| [deploy_bokeh_top_ALS_rating_proxy](./deploy_bokeh_top_ALS_rating_proxy) | Folder with Bokeh server based board game recommender web app, Top ALS Rating with User Proxies method |



