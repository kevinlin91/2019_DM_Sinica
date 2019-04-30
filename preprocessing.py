
# coding: utf-8

# <h3>Data Preprocessing</h3>
# <p>Data preprocessing is an important step in the data mining process<p>
# <ul>
#     <li>Data clean</li>Fill in missing values, smooth noisy data, identify or remove outliers, and resolve inconsistencies
#     <li>Data integration</li>Integration of multiple databases, data cubes, or files
#     <li>Data reduction</li>Obtains reduced representation in volume but produces the same or similar analytical results
# </ul>
# 

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('train.csv')
data


# In[3]:


data.shape
data.columns


# In[4]:


data.describe()


# In[5]:


for column in data.columns:
    print (column, data[column].isnull().any())


# In[6]:


import matplotlib.pyplot as plt
data.boxplot();


# In[7]:


data.hist();


# In[8]:


data['Age'].hist();


# In[9]:


data['Survived'].hist();


# In[10]:


data.groupby('Survived').hist();


# In[11]:


data['Sex'] = pd.factorize(data.Sex)[0]
data


# In[12]:


data.groupby('Survived').hist();


# In[13]:


from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde');


# In[14]:


data['Age'].fillna(29, inplace=True)
data

