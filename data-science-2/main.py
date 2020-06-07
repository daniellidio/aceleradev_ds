#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


# In[3]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


athletes = pd.read_csv("athletes.csv")


# In[5]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[6]:


athletes.head(5)


# In[7]:


athletes.tail(5)


# In[ ]:


alpha = 0.05


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[8]:


def q1():
    height_sample = get_sample(athletes, 'height', 3000)
    shapiro_height = sct.shapiro(get_sample(athletes, 'height', 3000))
    alpha = 0.05
    shapiro_height_is_normal = shapiro_height[1] > alpha
    return shapiro_height_is_normal


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[9]:


height_sample = get_sample(athletes, 'height', 3000)
shapiro_height = sct.shapiro(height_sample)


# In[10]:


plt.hist(height_sample, bins=25)


# In[11]:


qqplot(height_sample, line='s')


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[23]:


def q2():
    height_sample = get_sample(athletes, 'height', 3000)
    jarque_height = sct.jarque_bera(height_sample)
    alpha = 0.05
    jarque_height_is_normal = jarque_height[1] > alpha
    return bool(jarque_height_is_normal)


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[14]:


def q3():
    weight_sample = get_sample(athletes, 'weight', 3000)
    k2, p = sct.normaltest(weight_sample)
    alpha = 0.05
    normaltest_weight = p > alpha
    return bool(normaltest_weight)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[15]:


weight_sample = get_sample(athletes, 'weight', 3000)
plt.hist(weight_sample, bins=25)


# In[16]:


sns.boxplot(weight_sample)


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[17]:


def q4():
    log_weight = np.log(weight_sample)
    k2, p = sct.normaltest(log_weight)
    alpha = 0.05
    normaltest_log_weight = p > alpha
    return bool(normaltest_log_weight)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[18]:


log_weight = np.log(weight_sample)
plt.hist(log_weight, bins=25)


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[19]:


bra_athletes = athletes[athletes['nationality'] == 'BRA']
usa_athletes = athletes[athletes['nationality'] == 'USA']
can_athletes = athletes[athletes['nationality'] == 'CAN']


# In[20]:


def q5():
    ttest_bra_usa, p_bra_usa = sct.ttest_ind(bra_athletes['height'], usa_athletes['height'], equal_var=False, nan_policy='omit')
    alpha = 0.05
    return bool(p_bra_usa > alpha)


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[21]:


def q6():
    ttest_bra_can, p_bra_can = sct.ttest_ind(bra_athletes['height'], can_athletes['height'], equal_var=False, nan_policy='omit')
    alpha = 0.05
    return bool(p_bra_can > alpha)


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[22]:


def q7():
    ttest_usa_can, p_usa_can = sct.ttest_ind(usa_athletes['height'], can_athletes['height'], equal_var=False, nan_policy='omit')
    return round(float(p_usa_can), 8)


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
