#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('seaborn')
#Bibliotecas utilizadas

##Carrega o dataset
iris = datasets.load_iris() #Carrega o dataset na variável 'iris'
X = iris.data[:, :2] #Features 1 e 2
Y = iris.target #Espécies target

iris_df = pd.concat([pd.DataFrame(X),pd.DataFrame(Y)], axis=1) #Concatenação das features com os targets p/o dataframe
iris_df.columns = ['sepal_length', 'sepal_width', 'target'] #Nomeação das colunas
iris_df2 = iris_df[iris_df['target'] != 2] #Filtro de duas espécies
X = iris_df2.iloc[:, :2] #Atribuição das features para a variável X
Y = iris_df2.iloc[:, 2] #Atribuição do target para a variável y

##Atribuição dos valores -1 e 1 para as duas espécies
def troca(df): #Separação dos target em -1 e 1
    if df == 0:
        return -1
    else:
        return df
Y = Y.apply(troca) #Aplicação da função

##Plot das duas espécies
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=Y, cmap='plasma')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
#plt.savefig("Iris_sepal.png", dpi=120)

#Minimização por Lagrange
new_alfa = np.random.uniform(0,1,100) #Define os alfas iniciais por distribuição uniforme

from sklearn.preprocessing import StandardScaler #Normalização dos alfas iniciais
taxa = 0.0001 #Taxa de ajuste do gradiente
dalf = 0.01 #Incremento do alfa

scaler = StandardScaler() #Função que normaliza os alfas de forma que o somatório do produto alfa[i]*y[i] = 0
scaler.fit((new_alfa*Y.values).reshape(-1,1)) #Fita os dados
att = scaler.transform((new_alfa*Y.values).reshape(-1,1)) #Transforma os dados
att = att.reshape(100) #Reshape para operações posteriores
alfa = att/Y.values #Cálculo do alfa agora que o produto alfa*y respeita a condição

##Omitizações para encontrar o alfa
epoch = 100 #Número de otimizações

def der(X, dalfa): #Cálculo da derivada da função de Lagrange
    L_int = 0
    for i in range(len(X)):
        L_int += alfa[i] + dalfa - (1/2)*(alfa[i] + dalfa)*(alfa[j] + dalfa)*Y[i]*Y[j]*np.dot(X.iloc[i,:],X.iloc[j,:].T)
    return L_int
for i in range(epoch):
    j = np.random.randint(0,100) #Escolhe um dos alfas para ser iterado
    alfa[j] = alfa[j] - taxa*(der(X, dalf) - der(X, 0))/dalf #Cálculo da taxa
    scaler.fit((alfa*Y.values).reshape(-1,1)) #Renormalização do alfa
    att = scaler.transform((alfa*Y.values).reshape(-1,1)) #Transformação e reshape
    att = att.reshape(100)
    alfa = att/Y.values #Cálculo do alfa após normalização

w = np.zeros(2) #Criação de um vetor vazio para w
w[0] = (alfa*Y.values*X.iloc[:,0]).sum() #Cálculo da componente 'x' de w
w[1] = (alfa*Y.values*X.iloc[:,1]).sum() #Cálculo da componente 'y' de w

#print(w[0]) #Print para teste
#print(w[1]) #Print para teste

##Plot do resultado do algoritmo
plt.scatter(range(len(np.dot(w,X.values.T))), np.dot(w,X.values.T), c=Y, cmap='plasma')
plt.xlabel('Índice do ponto no dataset')
plt.ylabel('w . x')
#plt.savefig("w_x.png", dpi=120)