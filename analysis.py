

import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

# Carregar os dados de gender wage gap
genderwagegap_df = spark.read.csv('gender-wage-gap-by-occupation-annual.csv', header=True, inferSchema=True)

# Carregar os dados de female share of low pay earners
femaleshareoflowpayearners_df = spark.read.csv('female-share-of-low-pay-earners-annual.csv', header=True, inferSchema=True)

# Visualizar os primeiros registros dos dados de gender wage gap
genderwagegap_df.show()

# Visualizar os primeiros registros dos dados de sfemale share of low pay earners
femaleshareoflowpayearners_df.show()

# Converter os DataFrames Spark em DataFrames Pandas para facilitar a visualização
femaleshareoflowpayearners_pd = femaleshareoflowpayearners_df.toPandas()
genderwagegap_pd = genderwagegap_df.toPandas()

# Gráfico gender wage gap
plt.plot(genderwagegap_pd['ano'], genderwagegap_pd['gender wage gap'])
plt.xlabel('Ano')
plt.ylabel('gender wage gap')
plt.title('gender wage gap ao longo dos anos')
plt.show()

# Gráfico female share
plt.plot(femaleshareoflowpayearners_pd['ano'], femaleshareoflowpayearners_pd['salario_medio'])
plt.xlabel('Ano')
plt.ylabel('Female share of low pay earners')
plt.title('Female share of low pay earners ao longo dos anos')
plt.show()

# DataFrames Spark em DataFrames Pandas 
genderwagegap_pd = genderwagegap_df.join(femaleshareoflowpayearners_df, on='ano').toPandas()
correlation_matrix = genderwagegap_pd[['gender wage gap', 'salario_medio']].corr()
print(correlation_matrix)

import statsmodels.api as sm

# Regressão
X = genderwagegap_pd['gender wage gap']
Y = femaleshareoflowpayearners_pd['salario_medio']
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())

# Predição
genderwagegap_pred = 0.8  # Valor fictício
genderwagegap_pred = sm.add_constant(genderwagegap_pred)
femaleshare_pred = results.predict(genderwagegap_pred)
print('Salário médio previsto:', femaleshare_pred)
