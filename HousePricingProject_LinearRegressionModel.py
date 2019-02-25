#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[5]:


boston


# In[8]:


df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)


# In[9]:


df_x.describe()


# In[10]:


reg = linear_model.LinearRegression()


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size = 0.2, random_state=4)


# In[12]:


reg.fit(x_train,y_train)


# In[14]:


reg.coef_


# In[19]:


a = reg.predict(x_test)
a [1]


# In[18]:


y_test


# In[20]:


# mean square error 

np.mean((a-y_test)**2)


# In[ ]:




