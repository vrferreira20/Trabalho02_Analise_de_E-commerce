# Código de diagnóstico e execução segura
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import traceback

DATASET = 'Base Pesquisa Final F1 (t_reduzido).csv'

def try_read(path):
    # tenta leituras comuns que falham (sep, encoding)
    attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
    ]
    for a in attempts:
        try:
            df = pd.read_csv(path, sep=a["sep"], encoding=a["encoding"], engine='python')
            print(f"✔ Arquivo lido com sep='{a['sep']}' encoding='{a['encoding']}'")
            return df
        except Exception as e:
            print(f"⨯ Falhou com sep='{a['sep']}' encoding='{a['encoding']}': {e}")
    raise FileNotFoundError(f"Não foi possível ler o arquivo: {path}")

def diagnostics(df):
    print("\n--- SHAPE ---")
    print(df.shape)
    print("\n--- COLUNAS ---")
    print(list(df.columns))
    print("\n--- TIPOS ---")
    print(df.dtypes)
    print("\n--- 5 primeiras linhas ---")
    print(df.head().to_string(index=False))
    print("\n--- CONTAGEM NA/NAN por coluna ---")
    print(df.isna().sum())
    # verificar colunas-chave
    for col in ['Idade', 'Temp_med', 'Meio_pagamento', 'Renda_Men', 'Valor_med_gasto', 'Sexo']:
        if col in df.columns:
            print(f"\n-> amostra única/valores para '{col}':")
            print(df[col].dropna().unique()[:10])
        else:
            print(f"\n-> coluna '{col}' NÃO encontrada no dataset")

def safe_to_numeric(df, col):
    if col not in df.columns:
        raise KeyError(f"Coluna '{col}' não existe")
    # tentar converter, mostrar quantos valores falharam
    coerced = pd.to_numeric(df[col], errors='coerce')
    n_non_numeric = coerced.isna().sum() - df[col].isna().sum()
    print(f"Coluna '{col}': {n_non_numeric} valores convertidos para NaN por serem não-numéricos")
    return coerced

def run_all(path):
    try:
        df = try_read(path)
        diagnostics(df)

        # Conferir existência de colunas antes de plotar
        # substitua nomes se seu dataset usar nomes diferentes (ex: 'idade' minúsculo)
        required = ['Idade', 'Temp_med']
        for r in required:
            if r not in df.columns:
                raise KeyError(f"Coluna obrigatória ausente: '{r}'. Verifique o nome exato (sensível a maiúsculas).")

        # Converter para numérico (se necessário)
        df['Idade'] = safe_to_numeric(df, 'Idade')
        df['Temp_med'] = safe_to_numeric(df, 'Temp_med')

        # Remover linhas sem essas duas colunas
        before = df.shape[0]
        df = df.dropna(subset=['Idade', 'Temp_med'])
        after = df.shape[0]
        print(f"\nRemovidas {before - after} linhas com Idade/Temp_med faltantes.")

        # Agora plotar correlação (se houver colunas numéricas)
        try:
            corr = df.corr(numeric_only=True)
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title("Mapa de Correlação (diagnóstico)")
            plt.show()
        except Exception as e:
            print("Falha ao plotar heatmap:", e)

        # Regressão linear (Idade -> Temp_med)
        X = df[['Idade']]
        y = df['Temp_med']
        modelo = LinearRegression()
        modelo.fit(X, y)
        y_pred = modelo.predict(X)
        a = modelo.intercept_
        b = modelo.coef_[0]
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        print("\n--- Regressão Linear (Idade -> Temp_med) ---")
        print(f"Equação: Temp_med = {a:.4f} + ({b:.4f} × Idade)")
        print(f"R² = {r2:.4f}    MSE = {mse:.4f}")

        # Gráfico seguro
        plt.figure(figsize=(8,5))
        sns.scatterplot(x=X.squeeze(), y=y)
        sns.lineplot(x=X.squeeze(), y=y_pred)
        plt.title("Idade vs Temp_med — regressão")
        plt.xlabel("Idade")
        plt.ylabel("Temp_med")
        plt.show()

        print("\nExecutado com sucesso. Se houver warnings, copie/cole aqui para eu analisar.")
    except Exception as e:
        print("\n--- ERRO durante execução ---")
        traceback.print_exc()

# Execute diagnóstico
run_all(DATASET)
