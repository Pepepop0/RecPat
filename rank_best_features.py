import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Carregar o dataset
file_path = './updated_pollution_dataset.csv'  # Caminho do arquivo carregado
data = pd.read_csv(file_path)

# Definir as features e a variável alvo
X = data.drop(columns=['Air Quality'])  # Coluna 'Air Quality' como alvo
y = data['Air Quality']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline corrigida
pipeline = Pipeline([
    ('scaler', StandardScaler()),                    # Escalador
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),  # Seleção de features (f_classif)
    ('model', GaussianNB())                          # Modelo Naive Bayes
])

# Treinamento
pipeline.fit(X_train, y_train)

# Extrair o seletor de características e seus scores
selector = pipeline.named_steps['feature_selection']
scores = selector.scores_

# Criar um DataFrame para rankear as características
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': scores
})

# Ordenar por score de forma decrescente
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# Exibir as 5 características mais importantes
print(feature_scores.head(10))
