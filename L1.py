#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[4]:


df = pd.read_csv('Salary_Data.csv')
df.head(5)
#a = df['x'].to_numpy()
#b = df['y'].to_numpy()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(df.YearsExperience, df.Salary, color='green')


# In[6]:


#df.x = df.x.fillna(df.x.median())
#df.y = df.y.fillna(df.y.median())


# In[7]:


df


# In[8]:


reg = linear_model.LinearRegression()
reg.fit(df[['YearsExperience']], df.Salary)


# In[9]:


reg.intercept_


# In[10]:


reg.coef_


# In[11]:


reg.predict([[5.5]])


# In[ ]:




