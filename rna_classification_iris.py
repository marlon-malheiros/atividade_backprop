#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:33:27 2018

@author: marlon
"""

from random import seed, random
from math import exp
from datasets import dataset, datatest
import matplotlib.pyplot as plt

#Função de inicialização da rede
def inicia_rede(n_entradas, n_ocultos, n_saidas):
    """
    Recebe o número de entradas, número de neurônios na camada oculta e o número de neurônios na camada de saida. 
    Retorna uma rede(lista) com n_ocultos neuronios(cada neurônio é representado como um dicionário contendo todos os parâmetros desse neurônio), possuindo n_entradas+1(pesos e bias) valores inicialmente, e n_saidas neurônios na camada de saída, cada um contendo n_ocultos+1(pesos e bias)parâmetros, todos os parâmetros 
    são iniciados com valores aleatórios.
    Ex: rede = inicia_rede(2,1,2)
        o   o  
         \ /
          o    rede[0] = peso_11, peso_21, bias
         / \
        o   o  rede[1] = (peso_11, bias), (peso_12, bias)
        
        print(rede)
        [[{'pesos': [0.9136160737762681, 0.4348238063299966, 0.8514058849247363]}], neuronio da camada oculta
         [{'pesos': [0.34908179342226964, 0.3738186550989948]},   neuronio1 da camada de saida
         {'pesos': [0.32866068169686613, 0.7341867318530363]}]]   neuronio2 da camada de saida
    """
    rede = list()
    c_oculta = [ {'pesos': [random() for i in range(n_entradas+1)]} for i in range(n_ocultos)]
    rede.append(c_oculta)
    c_saida = [ {'pesos': [random() for i in range(n_ocultos+1)]} for i in range(n_saidas)]
    rede.append(c_saida)
    return rede

#Cálculo do valor passado para a função de ativação dos neurônios
def ativacao_neuronio(pesos, entradas):
    """
    Recebe os pesos do neurônio, e suas entradas, e efetua o calculo  de:
          ___
          \  
     z =  /__ w_j * i_j + b,
     
     e z será passado para função de ativação.  
    """
    ativacao = pesos[-1]
    for i in range(len(pesos)-1):
        ativacao += pesos[i] * entradas[i]
    return ativacao        

# Função de ativação
def transferencia_ativacao(ativacao):  
    """
    Recebe z calculado pela função ativacao_neuronio, e transfere para o neurônio para calcular a saída.
    """
    return 1.0 / (1.0 + exp(-ativacao))

# Derivada da função de ativação
def transf_ativacao_d(saida):
    """
    IMPORTANTE: Essa função sempre recebe o valor de saída da saída de um neurônio, ou seja transferencia_ativacao.
    """
    return saida * (1 - saida)

# Propagação de um sinal da entrada para a saida
def propagacao(rede, linha):
    """
    Recebe a rede atual, e uma linha contendo as entradas. E retorna saída da rede.
    """
    entradas = linha
    for camada in rede:
        novas_entradas = []
        for neuronio in camada:
            ativacao = ativacao_neuronio(neuronio['pesos'], entradas)
            neuronio['saida'] = transferencia_ativacao(ativacao)
            novas_entradas.append(neuronio['saida'])
        entradas = novas_entradas
    return entradas

# Retropropagacao de erro
def backpropagation(rede, esperado):
    """
    Recebe a rede, e o valor da saída esperada, e calcula os valores de delta para todos os neurônios da rede. Essa função só pode ser usada depois da função de propagação, pois esta calcula os valores das saídas dos neurônios.
    Erro na camada de saída = (valor_esperado - saida_rede) * Derivada_Funcao_De_Ativacao(saida_rede)
                                __
    Erro na camada oculta =     \
                                /__ w_j+1 * erro_j+1  * Derivada_Funcao_De_Ativacao(saida_rede_j)
    """
    for i in reversed(range(len(rede))):
        camada = rede[i]
        erros = list()
        if i != len(rede)-1:
            for j in range(len(camada)):
                erro = 0.0
                for neuronio in rede[i+1]:
                    erro += (neuronio['pesos'][j] * neuronio['delta'])
                erros.append(erro)
        else:
            for j in range(len(camada)):
                neuronio = camada[j]
                erros.append(esperado[j] - neuronio['saida'])
        for j in range(len(camada)):
            neuronio = camada[j]
            neuronio['delta'] = erros[j] * transf_ativacao_d(neuronio['saida'])
            
# Atualização de pesos
def atualizar_pesos(rede, linha, t_aprendizado):
    """
    Recebe a rede, uma linha de entrada e a taxa de aprendizado. E atualiza os pesos e os bias da rede. Essa função só pode ser chamada após a propagacao() e a backpropagation().
    """
    for i in range(len(rede)):
        entradas = linha[:-1]
        if i != 0:
            entradas = [neuronio['saida'] for neuronio in rede[i-1]]
        for neuronio in rede[i]:
            for j in range(len(entradas)):
                neuronio['pesos'][j] += t_aprendizado * neuronio['delta'] * entradas[j]
        neuronio['pesos'][-1] += t_aprendizado * neuronio['delta']
        
# Treino da rede
def treinar_rede(rede, dados_treino, t_aprendizado, n_epocas, n_saidas):
    """
    Recebe 1)Rede, 2)Matriz com os dados de treinamento, 3)Taxa de aprendizado, 4)Número de épocas de treinamento e 5)Número de saídas da rede.
    O método de treinamento usado foi o gradiente descendente estocástico.
    hot enconding
    INSIRA AQUI MAIS INFORMAÇÕES
    """
    l_erros = list()
    for epoca in range(n_epocas):                
        erro_quadrado = 0        
        for linha in dados_treino:
            saidas = propagacao(rede, linha)
            v_esperado = [0 for i in range(n_saidas)]           
            v_esperado[linha[-1]] = 1            
            erro_quadrado += sum([(v_esperado[i] - saidas[i])**2 for i in range(len(v_esperado))])            
            backpropagation(rede, v_esperado)
            atualizar_pesos(rede, linha, t_aprendizado)
        print('>Época=%d, T_aprendizado=%.3f, Erro=%.3f' % (epoca, t_aprendizado, erro_quadrado))
        #print('>Época=%d, T_aprendizado=%.3f, Erro² Médio=%.3f' % (epoca, t_aprendizado, erro_quadrado/n_epocas))
        l_erros.append(erro_quadrado)
    epoca_n = [i for i in range(n_epocas)]  
    plt.grid()      
    plt.plot(epoca_n, l_erros, linewidth=1.0)
    plt.ylabel('Erro Quadrático')
    plt.xlabel('Época')
    plt.title("Erro Quadrático")
    plt.show()
                        

# Predição de saida
def predicao(rede, linha):
    """
    Recebe 1) Rede TREINADA e 2)Uma linha da matriz de dados de teste, de onde se deseja prever a classe.
    
    """
    saidas = propagacao(rede, linha)
    return saidas.index(max(saidas))

# Normalização dos dados de entrada
def dataset_minmax(dados):
    """
    Recebe uma matriz de dados.
    Retorna uma lista contendo os valores mínimos e máximos de cada coluna.
    """
    stats = [[min(column), max(column)] for column in zip(*dados)]
    return stats

def normalizar(dados, minmax):
    """
    Recebe 1) Matriz de dados e 2)Matriz com os mínimos e máximos de cada coluna.
    Transforma os dados de treino para o range 0-1.
    """
    for linha in dados:
        for i in range(len(linha)-1):
            linha[i] = (linha[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    
######
seed()

################## DEFINA AQUI O CONJUNTO DE DADOS DE TREINO ##################
dataset = dataset
minmax = dataset_minmax(dataset)
normalizar(dataset, minmax)

################## DEFINA AQUI O CONJUNTO DE DADOS DE TESTE ##################
datatest = datatest
minmax_t = dataset_minmax(datatest)
normalizar(datatest, minmax)

################## O PROGRAMA PODE OBTER O NÚMERO DE ENTRADAS/SAIDAS AUTOMATICAMENTE ##################
n_entradas = len(dataset[0]) - 1
n_saidas = len(set([linha[-1] for linha in dataset]))

################## PARÂMETROS AJUSTÁVEIS PELO USUÁRIO ##################
t_aprendizado = 0.3
n_epocas = 30
n_ocultas = 3


################## ALGORITMO ##################
rede = inicia_rede(n_entradas, n_ocultas, n_saidas)
treinar_rede(rede, dataset, t_aprendizado, n_epocas, n_saidas)

################## TESTE OBTIDO/ESPERADO ##################
for linha in datatest:
    predito = predicao(rede, linha)
    print("Resultado Esperado:%d - Resultado obtido: %d" %( linha[-1], predito))    