import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Configurações adicionais
warnings.filterwarnings('ignore')

# Conectar ao banco de dados SQLite e carregar os dados de pressão e vazão, indicando impacto
conn = sqlite3.connect('dados_rupturas.db')
df_events = pd.read_sql_query('SELECT tempo, no, tubulacao, pressao, vazao, submetido_a_quebra FROM rupturas', conn)
conn.close()

# Verificar dados iniciais carregados
print("Dados iniciais:")
print(df_events.head())

# Pré-processamento dos dados
# Converter tempo para datetime e extrair características temporais
df_events['tempo'] = pd.to_datetime(df_events['tempo'])
df_events['hora'] = df_events['tempo'].dt.hour
df_events['dia_da_semana'] = df_events['tempo'].dt.dayofweek
df_events['mes'] = df_events['tempo'].dt.month
df_events.drop('tempo', axis=1, inplace=True)

# Transformar a coluna 'submetido_a_quebra' para 1/0
df_events['submetido_a_quebra'] = df_events['submetido_a_quebra'].astype(int)

# Codificar a coluna 'no' e 'tubulacao' usando LabelEncoder
le_no = LabelEncoder()
le_tubulacao = LabelEncoder()
df_events['no_encoded'] = le_no.fit_transform(df_events['no'])
df_events['tubulacao_encoded'] = le_tubulacao.fit_transform(df_events['tubulacao'])
df_events.drop(['no', 'tubulacao'], axis=1, inplace=True)

# Separar características (X) e rótulo (y)
X = df_events.drop('submetido_a_quebra', axis=1)
y = df_events['submetido_a_quebra']

# Realizar One-Hot Encoding nas colunas categóricas (hora, dia_da_semana, mes)
categorical_cols = ['hora', 'dia_da_semana', 'mes']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Dividir dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar escalonamento Min-Max
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Treinar o modelo
knn.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_knn = knn.predict(X_test_scaled)

# Avaliar o modelo
print("\nRelatório de Classificação para Detecção de Vazamentos e Rupturas:")
print(classification_report(y_test, y_pred_knn))

# Matriz de Confusão
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de Confusão - Detecção de Vazamentos e Rupturas")
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()

### Validação Cruzada com StratifiedKFold ###
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_accuracies = []

# Validação cruzada
for train_index, val_index in kf.split(X_train_scaled, y_train):
    X_fold_train, X_fold_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Treinar o modelo em cada fold
    knn.fit(X_fold_train, y_fold_train)
    
    # Avaliar no conjunto de validação
    scores = knn.score(X_fold_val, y_fold_val)
    knn_accuracies.append(scores)

print(f"\nAcurácias do KNN nos folds: {knn_accuracies}")
print(f"Acurácia média do KNN: {np.mean(knn_accuracies):.4f}")
