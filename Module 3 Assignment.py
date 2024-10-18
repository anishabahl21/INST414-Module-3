#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from sklearn.metrics import pairwise_distances


# In[25]:


credit_card_customers_df = pd.read_csv("C:/Users/bahla/Desktop/INST414/Credit Card Customer Data.csv")
    
customer_features = credit_card_customers_df[['Customer Key','Total_visits_bank','Total_visits_online', 'Total_calls_made']]

distances = pairwise_distances(customer_features)

customer_features.set_index('Customer Key', inplace = True)

print(customer_features)
    


# In[26]:


query_customer_key = 38414
query_customer_vector = customer_features.loc[query_customer_key].values.reshape(1,-1)
distances = pairwise_distances(query_customer_vector, customer_features.values)
distance_series = pd.Series(distances[0], index = customer_features.index)
top_similar_customers = distance_series.nsmallest(11).index[1:]

print(f"Top 10 similar customers to {query_customer_key}: ")
print(top_similar_customers)


# In[28]:


query_customer_key = 80655
query_customer_vector = customer_features.loc[query_customer_key].values.reshape(1,-1)
distances = pairwise_distances(query_customer_vector, customer_features.values)
distance_series = pd.Series(distances[0], index = customer_features.index)
top_similar_customers = distance_series.nsmallest(11).index[1:]

print(f"Top 10 similar customers to {query_customer_key}: ")
print(top_similar_customers)


# In[29]:


query_customer_key = 60732
query_customer_vector = customer_features.loc[query_customer_key].values.reshape(1,-1)
distances = pairwise_distances(query_customer_vector, customer_features.values)
distance_series = pd.Series(distances[0], index = customer_features.index)
top_similar_customers = distance_series.nsmallest(11).index[1:]

print(f"Top 10 similar customers to {query_customer_key}: ")
print(top_similar_customers)


# In[ ]:




