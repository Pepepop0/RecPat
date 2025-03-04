import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import OrdinalEncoder

# Carregar dataset
data = pd.read_csv('./updated_pollution_dataset.csv')
X = data.drop(columns=['Air Quality'])
y = data['Air Quality']
encoder = OrdinalEncoder()
y = encoder.fit_transform(y.values.reshape(-1, 1)).ravel()

# Configuração do k-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Definição dos modelos com hiperparâmetros ajustados
models = {
    "Naive Bayes": Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif, k=7)),
        ('model', GaussianNB(var_smoothing=0.00027387693197926166))
    ]),
    
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif, k=7)),
        ('model', SVC(C=75.05, gamma=0.0073, kernel='rbf'))
    ]),
    
    "MLP": Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif, k=7)),
        ('model', MLPClassifier(activation='tanh', alpha=0.0043, 
                                hidden_layer_sizes=(50,), solver='adam', random_state=1))
    ]),
    
    "Decision Tree": Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif, k=8)),
        ('model', DecisionTreeClassifier(ccp_alpha=0.0032, criterion='log_loss', 
                                         max_depth=13, min_samples_leaf=2, 
                                         min_samples_split=15, splitter='best', random_state=1))
    ])
}

# Armazenar os resultados
results = []

for model_name, model in models.items():
    accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(precision_score, average='macro')).mean()
    recall = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(recall_score, average='macro')).mean()
    
    results.append([model_name, accuracy, precision, recall])

# Criar DataFrame para visualizar os resultados
results_df = pd.DataFrame(results, columns=['Modelo', 'Acurácia', 'Precisão', 'Recall'])
print(results_df)
