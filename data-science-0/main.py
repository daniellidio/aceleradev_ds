#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[8]:


import pandas as pd
import numpy as np


# In[10]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[12]:


black_friday.head(5)


# In[14]:


black_friday.shape


# In[16]:


black_friday[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')].shape[0]


# black_friday('Gender')['User_ID'].nunique()

# In[18]:


black_friday.User_ID.nunique()


# In[20]:


black_friday.dtypes.nunique()


# In[22]:


black_friday.dropna().shape[0]


# In[24]:


(black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0]


# In[26]:


black_friday['Product_Category_3'].isnull().sum().max()


# In[28]:


black_friday.isnull().sum().max()


# In[30]:


black_friday['Product_Category_3'].value_counts()


# In[32]:


black_friday['Product_Category_3'].mode()[0]


# In[34]:


mean_purchase = black_friday.Purchase.mean()


# In[36]:


std_purchase = black_friday.Purchase.std()


# black_friday['Purchase_norm'] = (black_friday.Purchase - mean_purchase) / std_purchase

# black_friday['Purchase_norm']

# type(black_friday['Purchase_norm'].mean())

# black_friday[(black_friday['Purchase_norm'] >= -1) & (black_friday['Purchase_norm'] <=1)].shape[0]

# mean_purchase = black_friday.Purchase.mean()
# std_purchase = black_friday.Purchase.std()
# black_friday['Purchase_norm'] = (black_friday.Purchase - mean_purchase) / std_purchase
# return float(black_friday.Purchase_norm.mean())

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[48]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[50]:


def q2():
    return black_friday[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[52]:


def q3():
    return black_friday.User_ID.nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[54]:


def q4():
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[56]:


def q5():
    return (black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0]


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[57]:


def q6():
    return black_friday.isnull().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[58]:


def q7():
    return int(black_friday['Product_Category_3'].dropna().mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[59]:


def q8():
    return float(((black_friday['Purchase'] - black_friday['Purchase'].min()) / (black_friday['Purchase'].max() - black_friday['Purchase'].min())).mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[60]:


def q9():
    mean_purchase = black_friday.Purchase.mean()
    std_purchase = black_friday.Purchase.std()
    black_friday['Purchase_norm'] = (black_friday.Purchase - mean_purchase) / std_purchase
    return black_friday[(black_friday['Purchase_norm'] >= -1) & (black_friday['Purchase_norm'] <=1)].shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[61]:


def q10():
    return True

