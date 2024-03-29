#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[118]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[119]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[120]:


countries = pd.read_csv("countries.csv", thousands='.', decimal=',')


# In[121]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[122]:


countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[123]:


def q1():
    unique_countries = np.sort(countries['Region'].unique()).tolist()
    return unique_countries


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[124]:


def q2():
    discretize = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    intervals_pop = discretize.fit_transform(countries[['Pop_density']])
    return int((intervals_pop >= 9).sum())


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[125]:


def q3():
    encoded_reg_cli = pd.get_dummies(countries[['Region', 'Climate']].fillna('NaN'))
    return int(encoded_reg_cli.shape[1])


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[126]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[127]:


def q4():
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pipeline = make_pipeline(imputer, scaler)
    countries_num = countries.select_dtypes(include=[np.number])
    pipeline.fit(countries_num)
    transformed_test_country = pipeline.transform([test_country[2:]])
    arable = transformed_test_country[:, countries_num.columns.get_loc("Arable")]
    return float(np.around(arable.item(),3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[128]:


q25, q50, q75 = np.quantile(countries['Net_migration'].dropna(), [.25, .50, .75])

iqr = q75 - q25

countries_q5 = countries.copy()

countries_q5['Outlier'] = 0
countries_q5.loc[countries['Net_migration'] < (q25 - 1.5*iqr), 'Outlier'] = 1
countries_q5.loc[countries['Net_migration'] > (q75 + 1.5*iqr), 'Outlier'] = 1


# In[129]:


def q5():
    countries_q5['Upper_Outlier'] = 0
    countries_q5['Lower_Outlier'] = 0
    countries_q5.loc[countries_q5['Net_migration'] < (q25 - 1.5*iqr), 'Lower_Outlier'] = 1
    countries_q5.loc[countries_q5['Net_migration'] > (q75 + 1.5*iqr), 'Upper_Outlier'] = 1
    
    upper_len = len(countries_q5[countries_q5['Upper_Outlier'] == 1])
    lower_len = len(countries_q5[countries_q5['Lower_Outlier'] == 1])
    
    return (lower_len, upper_len, False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[130]:


from sklearn.datasets import fetch_20newsgroups

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[131]:


def q6():
    c_vec = CountVectorizer()
    counts = c_vec.fit_transform(newsgroups.data)
    return int(counts[:, c_vec.vocabulary_['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[132]:


def q7():
    v_vec = TfidfVectorizer().fit(newsgroups.data)
    tfidf = v_vec.transform(newsgroups.data)
    return float(tfidf[:, v_vec.vocabulary_['phone']].sum().round(3))

