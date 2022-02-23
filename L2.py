#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[41]:


df = pd.read_csv('train.csv')
df.head(5)


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(df.x, df.y, color='green')


# In[44]:


df.x = df.x.fillna(df.x.median())
df.y = df.y.fillna(df.y.median())


# In[45]:


df


# In[46]:


reg = linear_model.LinearRegression()
reg.fit(df[['x']], df.y)


# In[47]:


reg.intercept_


# In[48]:


reg.coef_


# In[51]:


test = pd.read_csv('test.csv')


# In[73]:


xtest = test['x']


# In[74]:


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# In[78]:


#xtest.to_numpy()
test


# In[84]:


for x in xtest:
    print(reg.predict([[x]]))
    


# In[85]:


for x in xtest:
    #print(xtest[x]+'='+reg.predict([[x]]))
    print('{}={}', format(xtest[x],reg.predict([[x]])))


# In[ ]:




