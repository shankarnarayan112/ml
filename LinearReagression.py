#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[25]:


df = pd.read_csv('train.csv')
df.head(5)
a = df['x'].to_numpy()
b = df['y'].to_numpy()


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(df.x, df.y, color='green')


# In[27]:


reg = linear_model.LinearRegression()
reg.fit(df[['x']], df.y)


# In[28]:


reg.predict([[20]])


# In[29]:


reg.coef_


# In[30]:


reg.intercept_


# In[ ]:




