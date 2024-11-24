import os
import pickle
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caminhos para salvar o modelo treinado e as colunas originais
model_path = 'random_forest_model.pkl'
columns_path = 'columns.pkl'

# Função para carregar e pré-processar os dados
def load_and_preprocess_data():
    # Conectar ao banco de dados
    conn = sqlite3.connect('dados_rupturas.db')
    df_events = pd.read_sql_query('SELECT tempo, no, tubulacao, pressao, vazao, submetido_a_quebra FROM rupturas', conn)
    conn.close()

    # Pré-processamento
    df_events['tempo'] = pd.to_datetime(df_events['tempo'])
    df_events['hora'] = df_events['tempo'].dt.hour
    df_events['dia_da_semana'] = df_events['tempo'].dt.dayofweek
    df_events['mes'] = df_events['tempo'].dt.month
    df_events.drop('tempo', axis=1, inplace=True)

    df_events['submetido_a_quebra'] = df_events['submetido_a_quebra'].astype(int)

    le_no = LabelEncoder()
    le_tubulacao = LabelEncoder()
    df_events['no_encoded'] = le_no.fit_transform(df_events['no'])
    df_events['tubulacao_encoded'] = le_tubulacao.fit_transform(df_events['tubulacao'])
    df_events.drop(['no', 'tubulacao'], axis=1, inplace=True)

    X = df_events.drop('submetido_a_quebra', axis=1)
    y = df_events['submetido_a_quebra']

    categorical_cols = ['hora', 'dia_da_semana', 'mes']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y

# Função para ajustar as colunas ao conjunto de treinamento
def align_columns(X, columns_path):
    if os.path.exists(columns_path):
        # Carregar as colunas originais do treinamento
        with open(columns_path, 'rb') as f:
            original_columns = pickle.load(f)

        # Adicionar colunas ausentes com valor 0
        for col in original_columns:
            if col not in X.columns:
                X[col] = 0

        # Remover colunas extras
        X = X[original_columns]
    else:
        # Salvar as colunas atuais como as originais
        with open(columns_path, 'wb') as f:
            pickle.dump(X.columns, f)
    
    return X

# Função para treinar e salvar o modelo
def train_and_save_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Escalonamento (opcional)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar o modelo Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Salvar o modelo e o escalador
    with open(model_path, 'wb') as f:
        pickle.dump({'model': rf, 'scaler': scaler}, f)

    print(f"Modelo treinado e salvo em: {model_path}")

    # Avaliação do modelo
    y_pred = rf.predict(X_test_scaled)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de Confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matriz de Confusão - Detecção de Vazamentos")
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

# Executar o script
if __name__ == '__main__':
    # Carregar e pré-processar os dados
    X, y = load_and_preprocess_data()

    # Garantir que as colunas estão alinhadas com o modelo original
    X = align_columns(X, columns_path)

    # Verificar se o modelo já existe
    if os.path.exists(model_path):
        print(f"Atualizando o modelo existente salvo em {model_path}...")
    else:
        print("Treinando um novo modelo...")
    
    train_and_save_model(X, y)
