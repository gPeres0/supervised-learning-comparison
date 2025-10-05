import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# Definir os nomes dos arquivos
DATA_FILE = './data/Dodgers_data.csv'
EVENTS_FILE = './data/Dodgers_events.csv'

# 1. Carregar e preparar o dataset principal (Dodgers.data)
# Os dados não têm cabeçalho, então definimos as colunas
df_data = pd.read_csv(DATA_FILE)


# 2. Tratamento de valores ausentes (Count = -1)
# O método é uma imputação iterativa baseada nos 4 vizinhos válidos mais próximos
# e arredondamento para cima.
def impute_missing_count(series):
    """
    Substitui -1 pela média aritmética dos 4 vizinhos válidos mais próximos,
    arredondando para cima. O valor imputado se torna válido para as 
    imputações subsequentes.
    """
    # Criar uma cópia para modificação
    imputed_series = series.copy().astype(float)
    
    # Encontrar todos os índices onde Count é -1
    missing_indices = imputed_series[imputed_series == -1].index.tolist()
    
    # Continuar o loop enquanto houver valores ausentes
    while missing_indices:
        i = missing_indices.pop(0) # Pega e remove o primeiro índice ausente
        
        # Encontrar os 4 vizinhos válidos mais próximos
        
        # Determinar os vizinhos anteriores
        valid_prev = []
        k = 1
        while len(valid_prev) < 2 and (i - k) >= 0:
            if imputed_series.iloc[i - k] != -1:
                valid_prev.append(imputed_series.iloc[i - k])
            k += 1

        # Determinar os vizinhos posteriores
        valid_next = []
        k = 1
        while len(valid_next) < 2 and (i + k) < len(imputed_series):
            if imputed_series.iloc[i + k] != -1:
                valid_next.append(imputed_series.iloc[i + k])
            k += 1
            
        # Combinar e pegar os 4 valores
        neighbors = valid_prev + valid_next
        
        # Se houver menos de 4 vizinhos (casos extremos no início/fim), 
        # a média é feita com os vizinhos disponíveis.
        if len(neighbors) == 0:
            # Não deve acontecer em um dataset real, mas é uma segurança.
            # Se não houver vizinhos, usa-se a média global válida.
            mean_val = imputed_series[imputed_series != -1].mean()
        else:
            mean_val = np.mean(neighbors)
        
        # Arredondar para cima (ceil) e converter para inteiro
        imputed_value = math.ceil(mean_val)
        
        # Substituir o valor e marcá-lo como imputado (válido para as próximas iterações)
        imputed_series.iloc[i] = int(imputed_value)
        
    return imputed_series.astype(int)

# Aplicar a função de imputação
df_data['Count_Imputed'] = impute_missing_count(df_data['Count'])
# Renomear Count_Imputed para Count para corresponder ao dataset final
df_data.drop('Count', axis=1, inplace=True)
df_data.rename(columns={'Count_Imputed': 'Count'}, inplace=True)


# 3. Adicionar a coluna "event" (0 ou 1)
# Carregar o arquivo de eventos
df_events = pd.read_csv(EVENTS_FILE)

# Adicionar a coluna 'event' ao dataset principal, inicializando com 0
df_data['Event'] = 0

# Iterar sobre os eventos e marcar as capturas de tráfego que caem no intervalo
for index, row in df_events.iterrows():
    date = pd.to_datetime(row['Date'])
    start = pd.to_datetime(row['Begin_Time'])
    end = pd.to_datetime(row['End_Time'])

    df_data.loc[(pd.to_datetime(df_data['Date']) == date) & (pd.to_datetime(df_data['Time']) >= start) & (pd.to_datetime(df_data['Time']) <= end), 'Event'] = 1
    print(df_data['Event'][index])


# 4. Formatação do Output como CSV

# Reordenar as colunas no formato final (Date, Time, Count, event)
df_final = df_data[['Date', 'Time', 'Count', 'Event']]

# O nome do arquivo de saída
OUTPUT_FILE = './data/Dodgers_processed.csv'

# Salvar o dataset atualizado
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"Pré-processamento concluído! O dataset atualizado foi salvo em '{OUTPUT_FILE}'.")
