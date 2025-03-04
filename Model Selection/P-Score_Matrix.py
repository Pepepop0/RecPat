from scipy.stats import ttest_rel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel

# Dataset carregado
file_path = './updated_pollution_dataset.csv'  
data = pd.read_csv(file_path)

# Definir as features e a variável alvo
X = data.drop(columns=['Air Quality'])  
y = data['Air Quality']

# Configuração do k-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Definir os modelos
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
        ('model', MLPClassifier(random_state=1, activation='tanh', alpha=0.0043, 
                            hidden_layer_sizes=50, solver='adam'))
    ]),
    
    "Decision Tree": Pipeline([
        ('scaler', StandardScaler()),                    
        ('feature_selection', SelectKBest(score_func=f_classif, k=8)),  
        ('model', DecisionTreeClassifier(ccp_alpha=0.0032, criterion='log_loss', 
                                         max_depth=13, min_samples_leaf=2, 
                                         min_samples_split=15, splitter='best'))
    ])
}

# Armazenar os resultados de cada modelo
results = {model_name: [] for model_name in models}

# Avaliar cada modelo usando k-fold cross-validation
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')  
    results[model_name] = scores  # Guarda os escores de cada modelo

# Converter os resultados para matriz numpy
model_names = list(results.keys())
num_models = len(model_names)
data_matrix = np.array(list(results.values()))

# Criar a matriz de p-values do Teste t de Student
p_values_matrix = np.zeros((num_models, num_models))

for i in range(num_models):
    for j in range(num_models):
        if i == j:
            p_values_matrix[i, j] = 1.0  # Comparação do modelo consigo mesmo
        else:
            stat, p_value = ttest_rel(data_matrix[i], data_matrix[j])  # Teste t pareado
            p_values_matrix[i, j] = p_value

# Criar DataFrame para visualização
p_values_df = pd.DataFrame(p_values_matrix, index=model_names, columns=model_names)

THRESHOLD = 0.05

# Criar máscara para destacar valores abaixo do threshold
highlight_mask = p_values_df < THRESHOLD

# Plotar matriz de p-values
plt.figure(figsize=(10, 8))  # Aumentar o tamanho da figura
sns.heatmap(
    p_values_df, 
    annot=True, 
    cmap="coolwarm", 
    fmt=".4f", 
    linewidths=0.5, 
    annot_kws={"size": 10}  # Ajusta o tamanho da fonte das anotações
)
plt.title("Matriz de p-values do Teste t de Student (Destaque para p < 0.05)", fontsize=12)
plt.xlabel("Modelo Comparado", fontsize=11)
plt.ylabel("Modelo Avaliado", fontsize=11)
#plt.show()

###############################################################################################

# Definir um threshold para significância estatística
threshold = 0.05

# Criar a matriz de p-values do Teste t de Student
p_values_matrix = np.zeros((num_models, num_models))

for i in range(num_models):
    for j in range(num_models):
        if i == j:
            p_values_matrix[i, j] = 1.0  # Comparação do modelo consigo mesmo
        else:
            stat, p_value = ttest_rel(data_matrix[i], data_matrix[j])  # Teste t pareado
            p_values_matrix[i, j] = p_value

# Criar DataFrame para visualização
p_values_df = pd.DataFrame(p_values_matrix, index=model_names, columns=model_names)

# Criar a matriz de bordas para destacar valores abaixo do threshold
highlight_mask = p_values_df < threshold

# Plotar matriz de p-values
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    p_values_df, 
    annot=True, 
    cmap="coolwarm", 
    fmt=".4f", 
    linewidths=0.5, 
    annot_kws={"size": 10},  # Ajustar tamanho da fonte das anotações
    cbar_kws={'label': 'p-value'}
)

# Aplicar bordas verdes nos valores significativos
for i in range(num_models):
    for j in range(num_models):
        if highlight_mask.iloc[i, j]:  # Se p < 0.05, adiciona um contorno verde
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=2))

# Adicionar títulos e rótulos
plt.title("Matriz de p-values do Teste t de Student (Destaque para p < 0.05)", fontsize=12)
plt.xlabel("Modelo Comparado", fontsize=11)
plt.ylabel("Modelo Avaliado", fontsize=11)

# Mostrar o gráfico
plt.show()

