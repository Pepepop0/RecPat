import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = './updated_pollution_dataset.csv'  
data = pd.read_csv(file_path)


X = data.drop(columns=['Air Quality'])  
y = data['Air Quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=8)),
    ('model', DecisionTreeClassifier( ccp_alpha= np.float64(0.0032135042552847537), class_weight= None, criterion= 'log_loss', max_depth= 13 , max_features= None, min_samples_leaf= 2, min_samples_split= 15, splitter='best'))
])

# Treinamento
pipeline.fit(X_train, y_train)

# Avaliação
y_pred = pipeline.predict(X_test)

# Resultados
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# Plotando a árvore de decisão
'''
t.figure(figsize=(20, 10))
plot_tree(pipeline.named_steps['model'], 
          feature_names=X.columns,  
          class_names=np.unique(y).astype(str), 
          filled=True, 
          rounded=True, 
          fontsize=6)
plt.title("Árvore de Decisão Treinada")
plt.show()
'''
