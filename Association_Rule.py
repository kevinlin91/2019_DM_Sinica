
# coding: utf-8

# <h1> Association Rule </h1>
# <h2> Apriori: https://github.com/rasbt/mlxtend </h2>
# <h2> Dataset from: https://www.kaggle.com/kaggle/recipe-ingredients-dataset </h2>

# In[1]:


import pandas as pd
import json


recipe_data = pd.read_json('./train.json')
print (recipe_data)


# In[2]:


ingredients = recipe_data['ingredients'].tolist()
print (ingredients[:5])


# In[3]:


from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori

oht = OnehotTransactions()
oht_ary = oht.fit(ingredients).transform(ingredients)
df_train = pd.DataFrame(oht_ary, columns=oht.columns_)
df_train


# In[4]:


frequent_itemsets_train = apriori(df_train, min_support=0.05, use_colnames=True)
print (frequent_itemsets_train)


# In[5]:


from mlxtend.frequent_patterns import association_rules

training_rules = association_rules(frequent_itemsets_train, metric="confidence", min_threshold=0.1)
training_rules

