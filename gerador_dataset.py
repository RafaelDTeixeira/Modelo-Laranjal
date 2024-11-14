# Importar bibliotecas necessárias
import wntr
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import random
import warnings
warnings.filterwarnings("ignore", message="Not all curves were used in")

# Define o caminho para o arquivo .inp original
original_inp_file = 'rede_laranjal_mapa_urbano.inp'
temp_inp_file = 'temp_inp_file.inp'

# Codificação adequada e salvar em UTF-8
with open(original_inp_file, 'r', encoding='latin-1') as f:
    inp_file_content = f.read()

# Arquivo temporário com codificação UTF-8
with open(temp_inp_file, 'w', encoding='utf-8', errors='replace') as temp_file:
    temp_file.write(inp_file_content)

# Configuração do banco de dados SQLite
db_path = 'dados_rupturas.db'  # Caminho do banco de dados SQLite

# Conectar ao banco de dados SQLite e criar tabela
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS rupturas (
        tempo TEXT,
        no TEXT,
        tubulacao TEXT,
        pressao REAL,
        vazao REAL,
        quebrado_ou_vazamento INTEGER,
        submetido_a_quebra INTEGER
    )
''')

# Parâmetros fixos de simulação
total_duration = 24 * 3600  # Duração total da simulação: 24 horas
minimum_pressure = 5.0       # Pressão mínima para PDD
required_pressure = 20.0      # Pressão necessária para PDD
min_pipe_diam = 0.3048        # Diâmetro mínimo da tubulação a ser incluída na análise (12 polegadas)

# Lista para armazenar todos os dados das simulações antes de salvar no banco de dados
all_simulation_data = []

# Função para executar uma simulação de ruptura e coletar dados
def simular_ruptura(wn, start_time, pipe_name):
    # Configurar controle de fechamento da tubulação
    pipe = wn.get_link(pipe_name)
    act = wntr.network.controls.ControlAction(pipe, 'status', 0)
    cond = wntr.network.controls.SimTimeCondition(wn, 'Above', start_time)
    ctrl = wntr.network.controls.Control(cond, act)
    wn.add_control('close pipe ' + pipe_name, ctrl)
    
    # Executar simulação de hidráulica com ruptura
    sim = wntr.sim.WNTRSimulator(wn)
    sim_results = sim.run_sim()

    # Coletar dados de pressão e vazão para todos os nós durante a simulação
    sim_pressure = sim_results.node['pressure']
    sim_demand = sim_results.node['demand']
    
    # Identificar nós impactados pela ruptura
    sim_pressure_below_pmin = sim_pressure.columns[(sim_pressure < minimum_pressure).any()]
    impacted_nodes = set(sim_pressure_below_pmin)

    # Calcular o fluxo médio na tubulação quebrada
    avg_flowrate = sim_results.link['flowrate'].loc[start_time::, pipe_name].mean()
    pipe_impact = 1 if avg_flowrate == 0 else 0  # Indicar impacto se o fluxo for zero
    
    # Armazenar dados da simulação na lista
    for time in sim_pressure.index:
        for node in sim_pressure.columns:
            pressure = sim_pressure.at[time, node]
            demand = sim_demand.at[time, node]
            quebrado_ou_vazamento = 1 if node in impacted_nodes else 0
            tempo_str = (datetime.now() + pd.to_timedelta(time, unit='s')).strftime('%Y-%m-%d %H:%M:%S')
            
            # Armazenar o ID da tubulação (pipe_name) no banco de dados
            all_simulation_data.append((tempo_str, node, pipe_name, pressure, demand, quebrado_ou_vazamento, pipe_impact))
        
        # Armazenar dados para a própria tubulação quebrada
        all_simulation_data.append((tempo_str, pipe_name, pipe_name, 0, avg_flowrate, 0, 1))

# Loop para definir o número de simulações e parâmetros aleatórios
num_simulacoes = 5  # Número de simulações desejado
for sim in range(num_simulacoes):
    print(f"Simulação {sim + 1}/{num_simulacoes}")

    # Recarregar o modelo para cada simulação
    wn = wntr.network.WaterNetworkModel(temp_inp_file)
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.time.duration = total_duration
    wn.options.hydraulic.minimum_pressure = minimum_pressure
    wn.options.hydraulic.required_pressure = required_pressure

    # Parâmetros aleatórios para a simulação
    num_pipes_to_break = random.randint(1, 5)  # Número aleatório de tubulações a serem quebradas
    pipes_to_break = random.sample(list(wn.query_link_attribute('diameter', np.greater_equal, min_pipe_diam).index), num_pipes_to_break)
    start_time = random.randint(1, 3) * 3600  # Tempo aleatório de início da ruptura (entre 1 e 3 horas)

    # Executar simulação para cada tubulação selecionada
    for pipe_name in pipes_to_break:
        try:
            simular_ruptura(wn, start_time, pipe_name)
        
        except Exception as e:
            print(f"Erro na simulação da tubulação {pipe_name}: {e}")

# Inserir todos os dados de uma vez no banco de dados
cursor.executemany('''
    INSERT INTO rupturas (tempo, no, tubulacao, pressao, vazao, quebrado_ou_vazamento, submetido_a_quebra)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', all_simulation_data)

# Confirmar inserções e fechar a conexão
conn.commit()
conn.close()

print("Dataset de rupturas salvo no banco de dados SQLite.")
