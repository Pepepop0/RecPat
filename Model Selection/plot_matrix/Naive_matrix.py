import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = './updated_pollution_dataset.csv' 
data = pd.read_csv(file_path)


X = data.drop(columns=['Air Quality']) 
y = data['Air Quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline corrigida
pipeline = Pipeline([
    ('scaler', StandardScaler()),                    
    ('feature_selection', SelectKBest(score_func=f_classif, k=7)),  
    ('model', GaussianNB(var_smoothing= np.float64(0.00027387693197926166)))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Naive Bayes")
plt.savefig("Matriz_confusão_Naive Bayes.png", dpi=300)
plt.show()

