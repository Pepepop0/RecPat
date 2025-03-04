import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    ('feature_selection', SelectKBest(score_func=f_classif, k=7)),
    ('model', MLPClassifier(random_state=1, activation='tanh', alpha=np.float64(0.004368108400282419), 
                            hidden_layer_sizes=50, learning_rate='constant', solver='adam'))
])

# Treinar e testar o modelo
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# Resultados
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - MLP")

# Salvar a matriz de confusão como imagem
plt.savefig("Matriz_confusão_MLP.png", dpi=300)
plt.show()
