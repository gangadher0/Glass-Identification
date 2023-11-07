#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('glass_data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('column_k' ,axis=1))
Scaled = scaler.transform(df.drop('column_k',axis=1))


# In[5]:


scaledview = pd.DataFrame(Scaled,columns=df.columns[:-1])
scaledview.head()


# In[6]:


# Train Test Split

from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(Scaled,df['column_k'],
                                                    test_size=0.20)


# In[10]:


import math
print(math .sqrt(len(y_train)))
print(len(y_train))


# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[13]:


knn = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')


# In[14]:


knn.fit(X_train,y_train)


# In[15]:


# Predictions and Evalutions

y_pre = knn.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score


# In[17]:


print(classification_report(y_test,y_pre))


# In[18]:


confusion_matrix(y_test,y_pre)


# In[19]:


accuracy_score(y_test,y_pre)


# In[20]:


residuals = y_test - y_pre
residuals.mean()
plt.scatter(y_pre, residuals)


# In[21]:


error_rate = []

for i in range(1,20):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[22]:


plt.figure(figsize=(10,6))
plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


# K = 11 is best

