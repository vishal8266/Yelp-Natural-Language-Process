
# coding: utf-8

# Natural Language Processing Project
# 
# Yelp Business Rating Prediction

# In[2]:

import numpy as np
import pandas as pd


# In[11]:

yelp = pd.read_csv('yelp.xls')


# In[12]:

yelp.head(2)


# In[13]:

yelp.info()


# In[14]:

yelp.describe()


# In[15]:

yelp['text length'] = yelp['text'].apply(len)


# Data Explore

# In[17]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().magic('matplotlib inline')


# In[20]:

g = sns.FacetGrid(yelp, col = 'stars')
g.map(plt.hist,'text length')


# In[22]:

sns.boxplot(x='stars',y='text length', data=yelp,palette='rainbow')


# In[24]:

stars = yelp.groupby('stars').mean()
stars


# In[25]:

stars.corr()


# In[28]:

sns.heatmap(stars.corr(),annot=True, cmap='coolwarm')


# Classification

# In[29]:

yelp_class = yelp[(yelp.stars==1) | (yelp.stars == 5)]


# In[31]:

x = yelp_class['text']
y = yelp_class['stars']


# In[32]:

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[33]:

X = cv.fit_transform(x)


# In[43]:

from sklearn.cross_validation import train_test_split


# In[44]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[45]:

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[46]:

nb.fit(X_train,y_train)


# Predictions

# In[47]:

predictions = nb.predict(X_test)


# In[48]:

from sklearn.metrics import confusion_matrix,classification_report


# In[51]:

print(confusion_matrix(y_test,predictions))
print ('\n')
print(classification_report(y_test,predictions))


# In[ ]:



