import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def loaddata(DATASET):
    data = None
    try:
        data = pd.read_csv(DATASET, sep=';')
        print("Dataset carregado com sucesso!")
    except:
        print("Load dataset Error!")
    return data

def DataTreat(data):
    #print(data.describe())
    corr = data.corr(numeric_only=True)
    #print(corr)

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa de Correlação entre Variáveis Numéricas")
    plt.show()

    return data

def barplot(data):
    sns.countplot(x="Meio_pagamento", palette="deep", data=data)
    plt.title("Distribuição dos meios de pagamento")
    plt.xlabel("Meio de Pagamento")
    plt.ylabel("Contagem")
    plt.show()

def barplot2(data):
    sns.countplot(x="Renda_Men", palette="deep", data=data)
    plt.title("Distribuição da renda")
    plt.xlabel("Renda")
    plt.ylabel("Contagem")
    plt.show()

def barplot3(data):
    sns.barplot(x="Idade", y="Temp_med", hue="Sexo", palette="deep", data=data)
    plt.title("Variação do Tempo Médio nas Redes Sociais (em horas)")
    plt.show()

def histoplot(data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x="Valor_med_gasto", multiple='stack', bins=60)
    # plt.yscale(value='linear')
    plt.title("Distribuição Dos Valores Gastos")
    plt.show()

def relpolot(data):
    sns.relplot(data, x='Temp_med', y='Idade')
    plt.title("Distribuição Do Tempo Médio pela Idade")
    plt.xlabel('Tempo médio de uso de redes')
    plt.ylabel('Idade')
    plt.show()

def data_idade(data):
    # Estatísticas descritivas do Idade
    media = data["Idade"].mean()
    moda = data["Idade"].mode()[0]
    mediana = data["Idade"].median()
    variancia = data["Idade"].var()
    desvio_padrao = data["Idade"].std()

    print(f"Estatistica da Idade")
    print(f"Media: {media:.2f}")
    print(f"Moda: {moda:.2f}")
    print(f"Mediana: {mediana:.2f}")
    print(f"Variancia: {variancia:.2f}")
    print(f"Desvio Padrao: {desvio_padrao:.2f}")

    return data

def data_idade(data):
    # Estatísticas descritivas do Idade
    media = data["Temp_med"].mean()
    moda = data["Temp_med"].mode()[0]
    mediana = data["Temp_med"].median()
    variancia = data["Temp_med"].var()
    desvio_padrao = data["Temp_med"].std()

    print(f"Estatistica do Tempo Médio de Uso das Redes")
    print(f"Media: {media:.2f}")
    print(f"Moda: {moda:.2f}")
    print(f"Mediana: {mediana:.2f}")
    print(f"Variancia: {variancia:.2f}")
    print(f"Desvio Padrao: {desvio_padrao:.2f}")

    return data

def data_idade(data):
    # Estatísticas descritivas do Idade
    media = data["Valor_med_gasto"].mean()
    moda = data["Valor_med_gasto"].mode()[0]
    mediana = data["Valor_med_gasto"].median()
    variancia = data["Valor_med_gasto"].var()
    desvio_padrao = data["Valor_med_gasto"].std()

    print(f"Estatistica do Valor Médio gasto Online")
    print(f"Media: {media:.2f}")
    print(f"Moda: {moda:.2f}")
    print(f"Mediana: {mediana:.2f}")
    print(f"Variancia: {variancia:.2f}")
    print(f"Desvio Padrao: {desvio_padrao:.2f}")

    return data

def linear_reg(data):
    # ------------------ 3. SELECIONAR VARIÁVEIS ------------------
    X = data[['Idade']]   # variável independente
    y = data['Temp_med']  # variável dependente

    # Remover possíveis valores ausentes
    data = data.dropna(subset=['Idade', 'Temp_med'])

    # ------------------ 4. TREINAR O MODELO ------------------
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Coeficientes da regressão
    a = modelo.intercept_
    b = modelo.coef_[0]

    print("\n Equacao da reta:")
    print(f"Temp_med = {a:.2f} + ({b:.2f} × Idade)")

    # ------------------ 5. FAZER PREVISÕES ------------------
    y_pred = modelo.predict(X)

    # ------------------ 6. MÉTRICAS DE AVALIAÇÃO ------------------
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print("\n Avaliacao do modelo:")
    print(f"R² (coeficiente de determinacao): {r2:.4f}")
    print(f"MSE (erro quadratico medio): {mse:.4f}")

    # ------------------ 7. VISUALIZAÇÃO ------------------
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=X.squeeze(), y=y, color='blue', label='Dados reais')
    sns.lineplot(x=X.squeeze(), y=y_pred, color='red', label='Regressão Linear')
    plt.title("Relação entre Idade e Tempo Médio de Uso das Redes")
    plt.xlabel("Idade")
    plt.ylabel("Tempo Médio de Uso das Redes")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------ 8. INTERPRETAÇÃO ------------------
    print("\n Interpretacao:")
    if b > 0:
        print(f"A cada aumento de 1 unidade em Idade, a Tempo Medio de Uso das Redes tende a aumentar {b:.2f} unidades, em media.")
    else:
        print(f"A cada aumento de 1 unidade em Idade, a Tempo Medio de Uso das Redes tende a diminuir {abs(b):.2f} unidades, em media.")
    print(f"O modelo explica aproximadamente {r2*100:.1f}% da variação em Tempo Medio de Uso das Redes com base na Idade.")


# ---------- Execução ----------
DATASET = 'Base Pesquisa Final F1 (t_reduzido).csv'
data = loaddata(DATASET)

if data is not None:
    data = DataTreat(data)
    #distribuicao_preco(data)
    #linear(data)
    #barplot(data)
    #barplot2(data)
    #barplot3(data)
    #histoplot(data)
    #relpolot(data)
    #data_idade(data)
    linear_reg(data)