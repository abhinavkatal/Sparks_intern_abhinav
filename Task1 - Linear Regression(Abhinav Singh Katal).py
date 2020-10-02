#!/usr/bin/env python
# coding: utf-8

# # Abhinav Singh Katal
# 

# # Task1 - Linear Regression

# In[3]:


# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Reading the data 
data = pd.read_csv('student.csv')


# In[5]:


#displaying data
data.head()


# In[15]:


# Plotting the data
data.plot(x = 'Hours', y = 'Scores')


# In[16]:


data.plot(x = 'Hours', y = 'Scores', style = 'o')


# In[88]:


# Storing indiviual columns into X and y
X = data.iloc[:,:-1].values
y = data.iloc[:, 1].values


# In[89]:


# Importing the train_test_split to split the data into training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[91]:


# Model creation
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, y_train)


# In[92]:


pred = linear.predict(X_test)
pred


# In[93]:


y_test


# In[94]:


data_new = pd.DataFrame({'Actual': y_test, 'Predicted' : pred})


# In[95]:


data_new


# In[100]:


#Predicting the query given in the email
score = linear.predict([[9.25]])


# In[101]:


score


# In[109]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,pred))


# # Hence the student will get approximately 93.69% if he/she studies for 9.25 hours

# In[106]:


#Plotting the line as given in the sample
#y=mx+C
#y = coefficient*X + Intercept
coefficient= linear.coef_
intercept = linear.intercept_


# In[108]:


plt.plot(X, coefficient*X + intercept)
plt.scatter(X,y)

