import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import time
#preparacion de dataset
df_clean=pd.read_csv('s3://tfm-davidaldomar/df_clean.csv')
df_clean=df_clean[df_clean['Price_Next_Year'].notna()]
df_clean=df_clean.fillna({'Dividends':0, 'Split':1})
cat_columns=df_clean[['Sector', 'Industry']]
num_columns=df_clean.drop(columns=['Sector', 'Industry','Name','Filename'])
#imputar datos categoricos
impute_cat = SimpleImputer(strategy='most_frequent')
df_cat = pd.DataFrame(impute_cat.fit_transform(cat_columns),columns = cat_columns.columns)
#imputar con mice
inicio = time.time()
imputer_mice = IterativeImputer(max_iter=100, random_state=8,verbose=2)
df_num_mice = pd.DataFrame(imputer_mice.fit_transform(num_columns),columns = num_columns.columns)
fin = time.time()
tiempo=fin-inicio
print(tiempo) 
with open("s3://tfm-davidaldomar/tiempo_mice.txt", "w") as output:
    output.write(str(tiempo))
data_mice = pd.concat([df_num_mice, pd.get_dummies(df_cat)], axis=1)
data_mice.to_csv(index=False,path_or_buf='s3://tfm-davidaldomar/data_mice.csv')
#imputar knn
inicio = time.time()
imputer_knn = KNNImputer(n_neighbors=5)
df_num_knn = pd.DataFrame(imputer_knn.fit_transform(num_columns),columns = num_columns.columns)
fin = time.time()
tiempo=fin-inicio
print(tiempo) 
with open("s3://tfm-davidaldomar/tiempo_knn.txt", "w") as output:
    output.write(str(tiempo))
data_knn = pd.concat([df_num_knn, pd.get_dummies(df_cat)], axis=1)
data_knn.to_csv(index=False,path_or_buf='s3://tfm-davidaldomar/data_knn.csv')




