# Contributing — NetMob 2026

## Guidelines

- **Use English** for all variable names, function names, and code comments
- **Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)** for commit messages  
  Examples: `feat: add weather preprocessing`, `fix: correct timezone conversion`, `docs: update EDA notes`
- **Mobility data is not in this repo** — the GPS telemetry files (~1.7 GB) and ticket data (~400 MB) are too large for GitHub and are under restricted access from the NetMob challenge. You already know how to access them locally.
- Open a Pull Request for every significant change — do not push directly to `main`

---

## Project Overview

**Theme:** Impact of weather conditions on passenger demand and system performance  
**Challenge:** [NetMob 2026 Data Challenge](https://netmob.org/www26/datachallenge.html)  
**Reference repo:** [lprm-ufes/Netmob2026](https://github.com/lprm-ufes/Netmob2026)

### Research Questions

This work investigates how different weather conditions influence passenger behaviour and the operational performance of the bus system in Niterói/RJ (March 2026). The study combines:

- Passenger boarding records (`ticket_data/`)
- Bus GPS telemetry (`mobility_data/`)
- Hourly meteorological measurements (`data/meteorological_data.csv`)

**Questions we aim to answer:**

1. Is there a significant variation in passenger volume on days with adverse weather compared to clear days?
2. Does the demand response to weather differ across passenger profiles (card type)?
3. During adverse weather, is there a measurable increase in headway or reduction in average operating speed?
4. Do bus lines show different levels of resilience to weather — are some more vulnerable than others?
5. Does the drop in bus frequency/speed during heavy rain amplify the fall in passengers, or does demand fall independently of service quality?

---

## Repository Structure

```
netmob2026/
├── data/
│   ├── meteorological_data.csv       # Hourly weather data — INMET station A001, March 2026
│   ├── mobility_data/                # ⚠️ Git-ignored — GPS telemetry (19 CSVs, 11–31/03, ~1.7 GB)
│   └── ticket_data/                  # ⚠️ Git-ignored — Boarding transactions (31 CSVs, 01–31/03, ~400 MB)
└── notebooks/
    ├── 01_eda_weather.ipynb          # Weather profile EDA + preprocessing
    ├── 02_eda_ticket.ipynb           # Passenger demand EDA + preprocessing
    └── 03_eda_mobility.ipynb         # Operational performance EDA + preprocessing
```

---

## Deliverables

### Entrega 0 — Project Proposal ✅
Initial objectives, research questions, team members, and data sources.  
→ See [objectives document](https://docs.google.com/document/d/1s7fWs9Qqgko_YvsTg5gYAqMJA3FnDEoUc3tzizF6aT4/edit)

---

### Entrega 1 — Preprocessing + EDA

**Goal:** Load all datasets, apply initial preprocessing, and perform exploratory data analysis aligned with the research questions.

Our analysis focuses on finding correlations between **weather conditions** (rain, temperature, wind) and **mobility problems** — both demand-side (fewer passengers) and supply-side (longer headways, lower speeds). To do this we need:

1. **Preprocessing** — clean and structure each dataset independently:
   - Handle nulls, fix timezones, create derived variables (daily rain totals, weather categories, `is_weekend` flag, etc.)
   - Validate data ranges and flag anomalies

2. **EDA** — explore distributions and correlations:
   - Weather profile: daily rain/temperature series, hourly heatmaps
   - Demand patterns: daily boarding volume controlled by weekday, breakdown by passenger profile and route
   - Operational performance: headway and speed distributions across weather categories
   - Cross-dataset: scatter plots and correlation coefficients (Pearson/Spearman) linking weather to demand and performance

> **Note:** The preprocessing and EDA steps are expected to evolve as analysis progresses. This first delivery establishes the foundation.

Notebooks for this delivery:
- `notebooks/01_eda_weather.ipynb`
- `notebooks/02_eda_ticket.ipynb`
- `notebooks/03_eda_mobility.ipynb`
