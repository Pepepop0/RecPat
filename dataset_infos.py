import pandas as pd

# Carregar o dataset
file_path = './updated_pollution_dataset.csv'  # Caminho para o arquivo CSV
data = pd.read_csv(file_path)

# Exibir as primeiras linhas do dataset
print(data.head())
# Estatísticas descritivas
print(data.describe())
# Verificar valores nulos
print(data.isnull().sum())


