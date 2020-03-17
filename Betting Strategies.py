#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# More Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

import os


# In[72]:


data_set = pd.read_csv("FootballDataEurope.csv")


# In[73]:


data_set.info()


# In[74]:


data_set.describe()


# In[61]:


data_set["country_name"].value_counts()


# In[71]:


data_set.head(1)
data_set["result"] = data_set.where(data_set["home_team_goal"] > data_set["away_team_goal"],"W","L")
#data_set["result"] = data_set.where(data_set["home_team_goal"] == data_set["away_team_goal"],"D",inplace=True)
#data_set["result"] = data_set.where(data_set["home_team_goal"] < data_set["away_team_goal"],"L",inplace=True)

#data_set["result"] = np.where(data_set["home_team_goal"] )
#data_set["result"] = np.where(data_set["home_team_goal"] == data_set["away_team_goal"],"D","W")


# In[56]:


#Creating data frames of countries which we will be focusing on
esp = data_set[data_set["country_name"]== "Spain"]
eng = data_set[data_set["country_name"]== "England"]
frc = data_set[data_set["country_name"]== "France"]
itl = data_set[data_set["country_name"]== "Italy"]


# In[57]:


eng


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




