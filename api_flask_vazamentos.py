from flask import Flask, request, jsonify
import pickle
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import LabelEncoder
from pyproj import Transformer  # Para conversão de coordenadas

# Configurações da API
app = Flask(__name__)
model_path = 'random_forest_model.pkl'
columns_path = 'columns.pkl'  # Caminho para as colunas usadas no treinamento

# Configuração do conversor de coordenadas
transformer = Transformer.from_crs("epsg:31982", "epsg:4326")  # EPSG:31982 para WGS84

# Função para carregar o modelo e o escalador
def load_model():
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']

# Função para buscar dados do banco SQLite
def query_database(query, params=()):
    conn = sqlite3.connect('dados_rupturas.db')
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# Função para alinhar as colunas do conjunto de dados ao modelo treinado
def align_columns(df):
    # Carregar as colunas usadas durante o treinamento
    with open(columns_path, 'rb') as f:
        trained_columns = pickle.load(f)

    # Adicionar colunas ausentes com valor 0
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    # Remover colunas extras que não estão no treinamento
    for col in df.columns:
        if col not in trained_columns:
            df.drop(columns=[col], inplace=True)

    # Reordenar as colunas para garantir a mesma ordem
    df = df[trained_columns]

    return df

def preprocess_data(df):
    # Converter a coluna 'tempo' para datetime, mantendo-a no DataFrame original
    df['tempo'] = pd.to_datetime(df['tempo'])

    # Criar colunas derivadas de 'tempo' para o modelo
    df['hora'] = df['tempo'].dt.hour
    df['dia_da_semana'] = df['tempo'].dt.dayofweek
    df['mes'] = df['tempo'].dt.month

    # Codificar colunas categóricas
    le_no = LabelEncoder()
    le_tubulacao = LabelEncoder()
    df['no_encoded'] = le_no.fit_transform(df['no'])
    df['tubulacao_encoded'] = le_tubulacao.fit_transform(df['tubulacao'])
    df.drop(['no', 'tubulacao'], axis=1, inplace=True)

    # One-Hot Encoding das colunas categóricas
    categorical_cols = ['hora', 'dia_da_semana', 'mes']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

# Função para converter coordenadas para latitude e longitude
def convert_to_latlon(x, y):
    lat, lon = transformer.transform(x, y)
    return lat, lon

@app.route('/vazamentos', methods=['GET'])
def detectar_vazamentos():
    # Carregar o modelo e o escalador
    model, scaler = load_model()

    # Recuperar parâmetros de entrada
    minutos = request.args.get('minutos', default=None, type=int)
    dia = request.args.get('dia', default=None, type=str)
    hora_inicio = request.args.get('hora_inicio', default=None, type=str)
    hora_fim = request.args.get('hora_fim', default=None, type=str)

    # Buscar dados do banco com base nos parâmetros
    if minutos:
        agora = datetime.now()
        tempo_limite = agora - timedelta(minutes=minutos)
        query = """
            SELECT tempo, no, tubulacao, pressao, vazao, x_coord, y_coord 
            FROM rupturas
            WHERE tempo BETWEEN ? AND ?
        """
        df = query_database(query, (tempo_limite.strftime('%Y-%m-%d %H:%M:%S'), agora.strftime('%Y-%m-%d %H:%M:%S')))
    elif dia:
        try:
            # Converter o dia em datetime
            dia_inicial = datetime.strptime(dia, '%Y-%m-%d')

            # Definir hora de início e fim
            if hora_inicio:
                dia_inicial = datetime.strptime(f"{dia} {hora_inicio}", '%Y-%m-%d %H:%M:%S')
            dia_final = dia_inicial + timedelta(days=1) if not hora_fim else datetime.strptime(f"{dia} {hora_fim}", '%Y-%m-%d %H:%M:%S')

            query = """
                SELECT tempo, no, tubulacao, pressao, vazao, x_coord, y_coord 
                FROM rupturas
                WHERE tempo >= ? AND tempo < ?
            """
            df = query_database(query, (dia_inicial.strftime('%Y-%m-%d %H:%M:%S'), dia_final.strftime('%Y-%m-%d %H:%M:%S')))
        except ValueError:
            return jsonify({'error': 'Formato inválido. Use YYYY-MM-DD para dia e HH:MM:SS para horas.'}), 400
    else:
        return jsonify({'error': 'Informe "minutos" ou "dia" como parâmetro.'}), 400

    # Verificar se existem dados
    if df.empty:
        return jsonify({'message': 'Nenhum dado encontrado para os parâmetros fornecidos.'}), 200

    # Pré-processar os dados
    df_processed = preprocess_data(df)

    # Alinhar as colunas do conjunto de dados
    df_processed = align_columns(df_processed)

    # Escalar os dados
    df_scaled = scaler.transform(df_processed)

    # Fazer previsões
    predictions = model.predict(df_scaled)

    # Adicionar resultados ao DataFrame
    df['submetido_a_quebra'] = predictions

    # Filtrar apenas rupturas detectadas
    df_vazamentos = df[df['submetido_a_quebra'] == 1]

    # Adicionar coluna 'hora_completa' e remover 'tempo'
    df_vazamentos['hora_completa'] = pd.to_datetime(df_vazamentos['tempo']).dt.strftime('%H:%M:%S')

    # Converter coordenadas para latitude e longitude
    df_vazamentos[['latitude', 'longitude']] = df_vazamentos.apply(
        lambda row: pd.Series(convert_to_latlon(row['x_coord'], row['y_coord'])), axis=1
    )

    df_vazamentos.drop(columns=['tempo', 'x_coord', 'y_coord'], inplace=True)

    # Retornar resultados como JSON
    resultados = df_vazamentos.to_dict(orient='records')
    return jsonify(resultados)

# Inicializar a API
if __name__ == '__main__':
    # Verificar se o modelo e as colunas existem
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        print("Erro: Modelo ou colunas não encontrados. Certifique-se de que o modelo foi treinado e salvo.")
    else:
        app.run(debug=True)


# Como usar:
# Busca por Vazamentos nos Últimos 30 Minutos:
# curl "http://127.0.0.1:5000/vazamentos?minutos=30"
# Busca por Vazamentos em um Dia Específico:
# curl "http://127.0.0.1:5000/vazamentos?dia=2024-11-23"
# Busca com Dia, Hora de Início e Fim:
# curl "http://127.0.0.1:5000/vazamentos?dia=2024-11-23&hora_inicio=14:00:00&hora_fim=18:00:00"