import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

tabela = pd.read_csv(r'C:\Users\Matheus\Desktop\Hash PY\Aula 4\a\advertising.csv')

y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(f'{r2_score(y_teste, previsao_regressaolinear):.2%}')
print(f'{r2_score(y_teste, previsao_arvoredecisao):.2%}')


nova_tabela = pd.read_csv(r'C:\Users\Matheus\Desktop\Hash PY\Aula 4\a\novos.csv')
print(nova_tabela)

previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)