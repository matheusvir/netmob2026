# Contribuindo — NetMob 2026

## Orientações gerais

- **Use inglês** para nomes de variáveis, funções e comentários no código
- **Siga o padrão [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)** para mensagens de commit  
  Exemplos: `feat: add weather preprocessing`, `fix: correct timezone conversion`, `docs: update EDA notes`
- **Os dados de mobilidade não estão neste repositório** — os arquivos de telemetria GPS (~1,7 GB) e os dados de ticket (~400 MB) são grandes demais para o GitHub e estão sob acesso restrito do desafio NetMob. Você já sabe como acessá-los localmente.
- Abra um Pull Request para toda mudança significativa — não faça push diretamente para `main`

---

## Visão geral do projeto

**Tema:** Impacto das condições climáticas na demanda de passageiros e no desempenho do sistema  
**Desafio:** [NetMob 2026 Data Challenge](https://netmob.org/www26/datachallenge.html)  
**Repositório de referência:** [lprm-ufes/Netmob2026](https://github.com/lprm-ufes/Netmob2026)

### Perguntas de pesquisa

Este trabalho investiga de que forma diferentes condições climáticas influenciam o comportamento dos passageiros e o desempenho operacional do sistema de ônibus de Niterói/RJ (março de 2026). O estudo combina:

- Registros de embarque de passageiros (`ticket_data/`)
- Telemetria GPS dos ônibus (`mobility_data/`)
- Medições meteorológicas horárias (`data/meteorological_data.csv`)

**Perguntas que queremos responder:**

1. Existe uma variação significativa no volume de passageiros associada a dias com condições climáticas adversas em comparação a dias com tempo firme?
2. O comportamento da demanda diante das variações climáticas difere entre os diversos perfis de passageiros do sistema?
3. Em períodos de tempo adverso, há um aumento perceptível nos intervalos entre os ônibus ou redução na velocidade média de operação?
4. As linhas de ônibus apresentam diferentes níveis de resiliência às condições do clima, havendo algumas mais vulneráveis que outras?
5. A redução na frequência e velocidade dos ônibus durante temporais potencializa a queda no número de passageiros, ou a redução da demanda ocorre independentemente da qualidade do serviço no momento?

---

## Estrutura do repositório

```
netmob2026/
├── data/
│   ├── meteorological_data.csv       # Dados climáticos horários — estação INMET A001, março/2026
│   ├── mobility_data/                # ⚠️ Ignorado pelo git — telemetria GPS (19 CSVs, 11–31/03, ~1,7 GB)
│   └── ticket_data/                  # ⚠️ Ignorado pelo git — transações de embarque (31 CSVs, 01–31/03, ~400 MB)
└── notebooks/
    ├── 01_eda_weather.ipynb          # EDA e pré-processamento dos dados climáticos
    ├── 02_eda_ticket.ipynb           # EDA e pré-processamento da demanda de passageiros
    └── 03_eda_mobility.ipynb         # EDA e pré-processamento do desempenho operacional
```

---

## Entregas

### Entrega 0 — Proposta inicial ✅
Objetivos, perguntas de pesquisa, membros da equipe e fontes de dados.  
→ Ver [documento de objetivos](https://docs.google.com/document/d/1s7fWs9Qqgko_YvsTg5gYAqMJA3FnDEoUc3tzizF6aT4/edit)

---

### Entrega 1 — Pré-processamento + EDA

**Objetivo:** Carregar todos os datasets, aplicar o pré-processamento inicial e realizar a análise exploratória alinhada às perguntas de pesquisa.

Nossa análise busca encontrar correlações entre **condições climáticas** (chuva, temperatura, vento) e **problemas de mobilidade e demanda** — tanto pelo lado da oferta (headways maiores, velocidades menores) quanto pelo lado da demanda (queda no número de passageiros). Para isso, são necessários dois passos:

1. **Pré-processamento** — limpar e estruturar cada dataset de forma independente:
   - Tratar nulos, corrigir timezones, criar variáveis derivadas (totais diários de chuva, categorias climáticas, flag `is_weekend`, etc.)
   - Validar intervalos de valores e sinalizar anomalias

2. **EDA** — explorar distribuições e correlações:
   - Perfil climático: série diária de chuva/temperatura, heatmaps horários
   - Padrões de demanda: volume diário de embarques controlado por dia da semana, breakdown por perfil de passageiro e por rota
   - Desempenho operacional: distribuições de headway e velocidade por categoria climática
   - Correlações cruzadas: scatter plots e coeficientes de correlação (Pearson/Spearman) ligando clima à demanda e ao desempenho

> **Nota:** O pré-processamento e a EDA estão sujeitos a ajustes conforme a análise avança. Esta entrega estabelece a base para as etapas seguintes.

Notebooks desta entrega:
- `notebooks/01_eda_weather.ipynb`
- `notebooks/02_eda_ticket.ipynb`
- `notebooks/03_eda_mobility.ipynb`
