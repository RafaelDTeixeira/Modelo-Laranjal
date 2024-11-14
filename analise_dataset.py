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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

# Configurações adicionais
warnings.filterwarnings('ignore')

# Caminho para salvar o modelo treinado
model_path = 'modelo_rupturas.h5'

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

# Aplicar SMOTE para balancear as classes no conjunto de treino
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Verificar a nova distribuição das classes após SMOTE
print("\nDistribuição das classes após SMOTE:")
print(pd.Series(y_train_res).value_counts())

### Carregar o modelo treinado existente, se houver ###
train_new_model = False
if os.path.exists(model_path):
    print("Modelo pré-existente encontrado.")
    update_model = input("Deseja atualizar o modelo existente? (s/n): ").strip().lower()
    if update_model == 's':
        train_new_model = True
    else:
        model = load_model(model_path)
        print("Modelo carregado para uso.")
else:
    train_new_model = True

### Treinamento do modelo, se necessário ###
if train_new_model:
    # Definir o modelo de rede neural
    model = Sequential([
        Dense(64, input_shape=(X_train_res.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Saída para classificação binária
    ])

    # Compilar o modelo
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinar o modelo com validação
    history = model.fit(X_train_res, y_train_res, epochs=100, batch_size=64,
                        validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

    # Salvar o modelo treinado
    model.save(model_path)
    print("Modelo treinado e salvo em:", model_path)

# Plotar as curvas de perda e acurácia
if train_new_model:
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia durante o treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda durante o treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

# Fazer previsões usando o modelo treinado
y_pred_nn_prob = model.predict(X_test_scaled)
y_pred_nn_class = (y_pred_nn_prob >= 0.5).astype(int)

# Avaliar o modelo
print("\nRelatório de Classificação para Detecção de Vazamentos e Rupturas:")
print(classification_report(y_test, y_pred_nn_class))

# Matriz de Confusão
conf_matrix_nn = confusion_matrix(y_test, y_pred_nn_class)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_nn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de Confusão - Detecção de Vazamentos e Rupturas")
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()

### Validação Cruzada com StratifiedKFold ###

# Definir o número de folds
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nn_accuracies = []

# Validação cruzada
for train_index, val_index in kf.split(X_train_res, y_train_res):
    X_fold_train, X_fold_val = X_train_res[train_index], X_train_res[val_index]
    y_fold_train, y_fold_val = y_train_res[train_index], y_train_res[val_index]

    # Definir o modelo para cada fold
    model_cv = Sequential([
        Dense(64, input_shape=(X_fold_train.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model_cv.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model_cv.fit(X_fold_train, y_fold_train, epochs=100, batch_size=64, verbose=0)
    
    # Avaliar no conjunto de validação
    scores = model_cv.evaluate(X_fold_val, y_fold_val, verbose=0)
    nn_accuracies.append(scores[1])

print(f"\nAcurácias da Rede Neural nos folds: {nn_accuracies}")
print(f"Acurácia média da Rede Neural: {np.mean(nn_accuracies):.4f}")
