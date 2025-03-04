import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, loguniform

# Carregar o dataset
data_path = "./updated_pollution_dataset.csv"
data = pd.read_csv(data_path)

# Definir variáveis de entrada (X) e saída (y)
x = data.drop(columns=["Air Quality"])
y = data["Air Quality"]

# Dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

# Criar o pipeline com pré-processamento e modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normaliza os dados
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),  # Seleciona as melhores features
    ('model', SVC(probability=True))  # Modelo SVM com probabilidade ativada
])

# Definir o espaço de busca para hiperparâmetros
param_dist = {
    'feature_selection__k': randint(3, 9),  # Número de features selecionadas
    'model__C': loguniform(0.001, 100),  # Regularização (evita overfitting e underfitting)
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Tipo de kernel
    'model__degree': randint(2, 5),  # Apenas para kernel='poly'
    'model__gamma': loguniform(0.0001, 1),  # Influência dos pontos (para kernel='rbf' e 'poly')
    'model__class_weight': [None, 'balanced']  # Ajuste para classes desbalanceadas
}

# Executar a busca de hiperparâmetros
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, 
    n_iter=5000, cv=5, scoring='accuracy', random_state=1, n_jobs=-1, verbose=2
)

random_search.fit(x_train, y_train)

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", random_search.best_params_)

# valiar o modelo no conjunto de teste
accuracy = random_search.score(x_test, y_test)
print(f"Taxa de acerto no teste: {accuracy:.4f}")
