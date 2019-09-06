#!/usr/bin/env python
# coding: utf-8

# # Springboard Capstone 1 helper functions
# 

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import time
import csv
import glob
import os.path
from os import path


# ### Text preprocessing function
# 
# This function prepares text data for training. For most models, the text will be processed further at training time, but pre-processing can save time when training is iterated multiple times.

# In[2]:


import re
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as sw

# function to prepare text for NLP analysis
def process_text(text, 
                 stemmer=None, 
                 regexstr=None, 
                 lowercase=True,
                 removestop=False,
                 verbose=True):
    """Helper function to pre-process text.
        Combines several preprocessing steps: lowercase, 
            remove stop, regex text cleaning, stemming.
        If savedpath is passed, then try to load saved processed text data
            and return that instead of processing."""
        
    if type(stemmer) == str:
        if stemmer.lower() == 'porter':
            stemmer = PorterStemmer()
        elif stemmer.lower() == 'snowball':
            stemmer = SnowballStemmer(language='english')
        else:
            stemmer = None
            
    # convert text list to pandas Series
    if type(text) == list or type(text) == np.array:
        processed = pd.Series(text)
    else:
        processed = text
    
    # make text lowercase
    if lowercase == True:
        if verbose: print('make text lowercase')
        processed = processed.str.lower()
        
    # remove stop words
    # NOTE: stop words w/ capitals not removed!
    if removestop == True:
        if verbose: print('remove stop words')
        stopwords = sw.words("english")
        processed = processed.map(lambda text: ' '.join([word for word in text.split() if word not in stopwords]))
        
    # apply regex expression
    if regexstr is not None:
        if verbose: print('apply regex expression')
        regex = re.compile(regexstr) 
        processed = processed.str.replace(regex,' ')
        
    # stemming
    # NOTE: stemming makes all lowercase
    if stemmer is not None:
        if verbose: print('stemming')
        processed = processed.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
        
    if verbose: print('done')
         
    return processed


# ## Load pre-processed text data
# 
# This function will do one of two things: 
# - If a specified csv file containing pre-processed text exists, then it will load it and return the contents as an array of text.
# - If the specified file does not exist, it will pre-process the given text, save that to the file, and return the pre-processed text.
# 
# The purpose of this function is to cut down on the time spent processing text data by storing pre-processed text (given specified processing parameters) to a file for retrieval the next time it is needed.
# 

# In[1]:


def load_processed_text(text, pathstart, processkwargs):
    """If possible, load a csv file containing pre-processed text data,
        otherwise process the text and save it to file for later loading."""
    
    # create name for prepared text file
    preppath = pathstart+' stem=%s, regex=%s, lower=%d, stop=%d.csv'%(
        processkwargs['stemmer'], 
        'None' if processkwargs['regexstr'] is None else processkwargs['regexstr'].replace('\\','(sl)'), 
        processkwargs['lowercase'], 
        processkwargs['removestop'])
    
    # if prepared text file exists, just load it
    if path.exists(preppath):
#         print('reading prepped text file: ', preppath)
        df = pd.read_csv(preppath, skip_blank_lines=False, na_filter=False)
        
    # file doesn't exist, so do the text prep and save the result for later loading
    else:
#         print('no prepped file, so reading original and prepping text...')
        text = process_text(text, **processkwargs)
        # create df
        df = pd.DataFrame({'text':text})
        # save prepared text file
        df.to_csv(preppath, index=False, na_rep='NaN')
        
    # return the processed text data
    return df['text'].values


# ### Load the feature data.
# 
# The comment data used in this analysis was prepared in three stages:
# 
# - [acquired using Reddit Python API PRAW](https://github.com/johnmburt/springboard/blob/master/capstone_1/reddit_collect_comments_v1.ipynb) from 12 subs. 8 of the subs are non-political, and 4 are political in nature. Models are trained on data for only one subreddit at a time, so that they are specialized to that subreddit.
# 
# 
# - The raw comment metadata was [processed using PCA to produce a single toxicity score](https://github.com/johnmburt/springboard/blob/master/capstone_1/reddit_generate_PCA_score_v2.ipynb) based on the votes and number of replies. Toxicity score was calculated and normalized within each subreddit and then ranged between -5 and +5 to create a toxicity score comparable between subs. The toxicity score was then thresholded to generate binary "toxic" vs. "not toxic" labels for supervised model training. The threshold applied was: score <= -1 = "toxic", otherwise "not toxic". 
# 
# 
# - [Features for training the models were generated](https://github.com/johnmburt/springboard/blob/master/capstone_1/reddit_comment_create_model_features_v1.ipynb) and saved to two sample aligned feature files for each subreddit. These files are used by the models for input.
# 
# **Note** This code is a little complicated because 1) for efficiency I'm using cached pre-processed text, and 2) sometimes after loading and processing there are samples with NaN values. To keep data aligned, I need to combine the two data dfs, remove the NaN samples, then split the df back into base and numeric and return those.
# 

# In[ ]:


def load_feature_data(subnames, srcdir, toxic_thresh=-1, text_prep_args=None):
    """Load and prep the feature data from two matched data files"""
    
    # load all data csvs for listed subs into dataframes 
    base_dfs = []
    numeric_dfs = []
    for sub in subnames:
        base_dfs.append(pd.read_csv(srcdir+'features_text_'+sub+'.csv'))
        numeric_dfs.append(pd.read_csv(srcdir+'features_doc2vec_'+sub+'.csv'))
        
    # concat all sub dfs into one for each data type
    base_df = pd.concat(base_dfs, ignore_index=True)
    numeric_df = pd.concat(numeric_dfs, ignore_index=True)
    
    # combine both dfs
    df = pd.concat([base_df, numeric_df], axis=1, ignore_index=True)
    df.columns = list(base_df.columns.values) + list(numeric_df.columns.values)
    
    # remove any columns with all nans
    df.dropna(axis=1, how='all', inplace=True)
    
    # remove any samples with nans
    df.dropna(inplace=True)
    
    # pre-process text 
    if text_prep_args is not None:
        pathstart = srcdir + 'processed_text_' + '_'.join(subnames)
        df['text'] = load_processed_text(df['text'], pathstart, text_prep_args)  
        
    # remove any samples with nans
    df.dropna(inplace=True)
    
    # split dfs back into text and numeric features
    base_df = df[base_df.columns]
    numeric_df = df[numeric_df.columns]
    
    # add numeric metadata features from base df to numeric df
    numeric_df['u_comment_karma'] = base_df['u_comment_karma']

    # return base df (text and all comment metadata), numeric features, training label
    return base_df['text'], numeric_df, np.where(base_df['pca_score']>toxic_thresh,0,1)


# ## Balance sample frequencies in training samples
# 
# The classifier may require balancing of sample frequencies between classes for best results. This function will up-sample to the specified number of samples per class.
# 
# The balance_classes_sparse function does sample balancing with sparse matrices, such as vectorized BOW data.

# In[ ]:


from scipy.sparse import vstack, hstack
from scipy.sparse.csr import csr_matrix

def balance_classes_sparse(X, y, samples_per_class=None, verbose=False):
    """Equalize number of samples so that all classes have equal numbers of samples.
    If samples_per_class==None, then upsample (randomly repeat) all classes to the largest class,
      Otherwise, set samples for all classes to samples_per_class."""
    
    def get_samples(arr, numsamples):
        if arr.shape[0] >= numsamples:
            index = np.arange(arr.shape[0])
            np.random.shuffle(index)
            return arr[index[:numsamples],:]
        else:
            samples = arr.copy()
            numrepeats = int(numsamples / arr.shape[0])
            lastsize = numsamples % arr.shape[0]
            for i in range(numrepeats-1):
                samples = vstack([samples,arr])
            if lastsize > 0:
                index = np.arange(arr.shape[0])
                np.random.shuffle(index)
                samples = vstack([samples, arr[index[:lastsize],:]])
            return samples   
    
    if verbose: 
        print('Balancing class sample frequencies:')
        
    # all class IDs
    classes =  pd.unique(y)
    classes = classes[~np.isnan(classes)]
    
    # get class with max samples
    if verbose: 
        print('\tOriginal sample frequencies:')
    if samples_per_class is None:
        samples_per_class = 0
        for c in classes:
            if verbose: 
                print('\t\tclass:',c,'#samples:',(np.sum(y==c)))
            samples_per_class = np.max([samples_per_class, np.sum(y==c)])
    if verbose: 
        print('\tNew samples_per_class:',samples_per_class)
                              
    # combine X and y
    Xy = csr_matrix(hstack([X, csr_matrix(np.reshape(y, (-1, 1)))]))
       
    # create a list of samples for each class with equal sample numbers 
    newdata = None
    for c in classes:
        if newdata is None:
            newdata = get_samples(Xy[y==c,:], samples_per_class)
        else:
            newdata = vstack([newdata, get_samples(Xy[y==c,:], samples_per_class)])
            
    if verbose:
        print('\ttotal balanced samples:',newdata.shape[0])
            
    return newdata[:,:-1], newdata[:,-1].toarray()


# ## Log results of each model test
# 
# This function logs the results of a model test to a CSV logfile. Every model notebook logs to the same file so that results can be compared.

# In[ ]:


import csv
import os.path
from os import path
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
from time import time

def log_model_results(logpath, modelname, subname, y_test, y_pred):
    """Write to CSV log file containing results of model train/test runs"""
    
    # write the header labels
    if not os.path.exists(logpath):
        labels = (['date','model','sub','num_nontoxic','num_toxic',
                   'acc_nontoxic','acc_toxic','accuracy','precision',
                   'recall','balanced_acc','F1','roc_auc'])
        with open(logpath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(labels)
            
    # create data row
    row = [datetime.datetime.now().strftime('%y%m%d_%H%M%S'),
          modelname, subname, 
          (y_test==0).sum(), (y_test==1).sum(),
           '%1.3f'%(((y_test==0) & (y_test==y_pred)).sum()/(y_test==0).sum()),
           '%1.3f'%(((y_test==1) & (y_test==y_pred)).sum()/(y_test==1).sum()),
           '%1.3f'%(np.sum((y_pred==y_test))/y_test.shape[0]),
           '%1.3f'%(precision_score(y_test, y_pred)),
           '%1.3f'%(recall_score(y_test, y_pred)),
           '%1.3f'%(balanced_accuracy_score(y_test, y_pred)),
           '%1.3f'%(f1_score(y_test, y_pred)),
           '%1.3f'%(roc_auc_score(y_test, y_pred))
          ]
    # write the data row
    with open(logpath, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(row)

