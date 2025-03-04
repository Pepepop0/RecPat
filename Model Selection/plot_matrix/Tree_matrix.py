import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o dataset
file_path = './updated_pollution_dataset.csv'
data = pd.read_csv(file_path)

# Definir as features e a variável alvo
X = data.drop(columns=['Air Quality'])
y = data['Air Quality']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline corrigida
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=8)),
    ('model', DecisionTreeClassifier( ccp_alpha= np.float64(0.0032135042552847537), class_weight= None, criterion= 'log_loss', max_depth= 13 , max_features= None, min_samples_leaf= 2, min_samples_split= 15, splitter='best'))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

# Resultados
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Tree")

# Salvar a matriz de confusão como imagem
plt.savefig("Matriz_confusão_Tree.png", dpi=300)
plt.show()

