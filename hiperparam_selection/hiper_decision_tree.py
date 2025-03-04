import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from scipy.stats import randint, uniform



data_path = "./updated_pollution_dataset.csv"
data = pd.read_csv(data_path)
x = data.drop(columns=["Air Quality"])
y = data["Air Quality"]
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),  
    ('model', DecisionTreeClassifier(class_weight="balanced", random_state=1))
])

# Definir os hiperparâmetros para busca
param_dist = {
    'feature_selection__k': randint(3, x.shape[1]),
    'model__criterion': ['gini', 'entropy', 'log_loss'],
    'model__max_depth': randint(3, 30),
    'model__min_samples_split': randint(5, 50),
    'model__min_samples_leaf': randint(2, 30),
    'model__splitter': ['best', 'random'],
    'model__ccp_alpha': uniform(0.0001, 0.02),
    'model__max_features': ['auto', 'sqrt', 'log2', None],
    'model__class_weight': [None, 'balanced']
}

# Executar RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist,
    n_iter=10_000, cv=3, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42
)

random_search.fit(x_train, y_train)


print("Melhores hiperparâmetros:", random_search.best_params_)
best_model = random_search.best_estimator_
print(f"Taxa de acerto no teste: {best_model.score(x_test, y_test):.4f}")
