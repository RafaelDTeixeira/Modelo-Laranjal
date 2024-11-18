# Modelo Hidráulico do Bairro Laranjal - Detecção de Vazamentos

Este projeto implementa um modelo hidráulico da rede de distribuição de água do bairro Laranjal, em Pelotas, utilizando um simulador para gerar datasets e um algoritmo para detectar rupturas e vazamentos nas tubulações. O modelo permite identificar vazamentos a partir dos dados de pressão e vazão, facilitando a análise e a manutenção da rede.

## Estrutura do Projeto

- **`rede_laranjal_mapa_urbano_18_06_24.inp`**: Arquivo de entrada contendo a configuração do modelo hidráulico, com a disposição de nós e tubulações para simulações.
- **`gerador_dataset.py`**: Script para gerar um dataset de eventos de ruptura e vazamentos simulados. Armazena os dados no banco de dados SQLite `dados_rupturas.db`.
- **`analise_dataset.py`**: Script para análise e treinamento de um modelo de rede neural para detecção de vazamentos, utilizando o arquivo de treinamento `modelo_rupturas.h5`.

## Scripts e Componentes

### Gerador de Dataset (`gerador_dataset.py`)
- Utiliza o simulador **WNTR** para simular eventos de ruptura e gerar dados de pressão e vazão em diferentes nós da rede.
- Realiza múltiplas simulações configurando rupturas em diferentes tubulações e horários aleatórios.
- Armazena os dados gerados no banco de dados SQLite `dados_rupturas.db`.
- Link do Dataset para teste: https://drive.google.com/file/d/1OPa4qvw17u95eBYx0rJGDxWf7Gf7PYMN/view?usp=sharing.

### Análise e Treinamento do Modelo (`analise_dataset.py`)
- Carrega e processa o dataset gerado, aplicando transformações temporais e codificação de variáveis categóricas.
- Treina uma rede neural para identificar rupturas na rede usando os dados simulados, armazenando o modelo treinado em `modelo_rupturas.h5`.
- Permite atualizar o modelo com novos dados ou realizar validações com *cross-validation* para avaliação de desempenho.

## Dependências

- Python 3.x
- Pacotes:
  - `wntr`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tensorflow`
  - `sklearn`
  - `sqlite3`

## Execução

1. **Gerar Dataset**: Execute `gerador_dataset.py` para simular rupturas e armazenar os dados no banco de dados.
2. **Treinar e Analisar Modelo**: Execute `analise_dataset.py` para treinar ou carregar o modelo de detecção de vazamentos.
 
## Contato

Este projeto é mantido por Rafael Teixeira. 

Para dúvidas ou sugestões, entre em contato pelo email: rafael.dteixeira@sou.ucpel.edu.br.
