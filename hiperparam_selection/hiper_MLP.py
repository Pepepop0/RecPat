import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# Carregar o dataset
data_path = "./updated_pollution_dataset.csv"
data = pd.read_csv(data_path)

# Separar variáveis dependentes e independentes
x = data.drop(columns=["Air Quality"])
y = data["Air Quality"]
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

# Criar o pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),
    ('model', MLPClassifier(max_iter=1000, random_state=1))
])

# Definir os hiperparâmetros para busca
param_dist = {
    'feature_selection__k': randint(3, x.shape[1]),
    'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam', 'sgd'],
    'model__alpha': uniform(0.0001, 0.01),  # Regularização L2
    'model__learning_rate': ['constant', 'adaptive']
}

# Executar RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist,
    n_iter=100, cv=3, scoring='accuracy', n_jobs=-1, verbose=2, random_state=1
)

random_search.fit(x_train, y_train)

# Mostrar os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", random_search.best_params_)

# Avaliar o modelo otimizado no conjunto de teste
best_model = random_search.best_estimator_
print(f"Taxa de acerto no teste: {best_model.score(x_test, y_test):.4f}")
