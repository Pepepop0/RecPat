import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, loguniform

data_path = "./updated_pollution_dataset.csv"
data = pd.read_csv(data_path)
x = data.drop(columns=["Air Quality"])
y = data["Air Quality"]
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)), 
    ('model', SVC(probability=True)) 
])

param_dist = {
    'feature_selection__k': randint(3, 9),
    'model__C': loguniform(0.001, 100),
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'model__degree': randint(2, 5), 
    'model__gamma': loguniform(0.0001, 1),  
    'model__class_weight': [None, 'balanced'] 
}

random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, 
    n_iter=5000, cv=5, scoring='accuracy', random_state=1, n_jobs=-1, verbose=2
)

random_search.fit(x_train, y_train)
print("Melhores hiperparâmetros:", random_search.best_params_)
accuracy = random_search.score(x_test, y_test)
print(f"Taxa de acerto no teste: {accuracy:.4f}")
