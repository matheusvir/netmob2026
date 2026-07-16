# NetMob 2026 — Final Analysis Delivery

This repository contains the analytical pipeline, exploratory notebooks, and final analyses developed for the NetMob 2026 Data Challenge. The project investigates how weather conditions affect passenger demand and bus system performance in Niterói/RJ during March 2026.

## Project goal

The work combines weather observations, passenger ticketing records, and bus mobility data to answer five research questions about:

1. demand variation under adverse weather;
2. differences across passenger profiles;
3. operational degradation during bad weather;
4. route-level resilience differences;
5. the joint effect of service quality and climate on demand.

## Repository structure

- `data/`
  - weather files versioned in the repository;
  - large raw mobility and ticket datasets are not stored in GitHub.
- `notebooks/`
  - `00_build_preprocessed_datasets.ipynb`: builds derived datasets from the raw inputs;
  - `01_eda_weather.ipynb`: weather exploratory analysis and preprocessing foundation;
  - `02_eda_ticket.ipynb`: passenger demand exploratory analysis and preprocessing foundation;
  - `03_eda_mobility.ipynb`: bus mobility exploratory analysis and preprocessing foundation;
  - `04_build_analytic_base.ipynb`: integrated analytic-base audit;
  - `05_analysis_demand_weather.ipynb`: demand vs weather analysis;
  - `06_analysis_service_vs_weather_gtfs.ipynb`: service performance vs weather with GTFS context;
  - `07_analysis_route_resilience.ipynb`: route resilience analysis;
  - `08_analysis_joint_demand_service.ipynb`: joint demand and service analysis;
  - `09_storytelling_outputs.ipynb`: final storytelling outputs for synthesis and presentation;
  - `10_final_narrative_index.ipynb`: central narrative notebook and recommended entry point for evaluators.
- `scripts/`
  - helper scripts for notebook generation and analysis support.
- `requirements-analysis.txt`
  - Python dependencies for the analytical environment.

## External data access

The complete raw datasets are too large to be stored in this GitHub repository. For that reason, the repository includes the notebooks, code, and documentation, while the raw challenge data must be downloaded separately.

## Reproducibility note

This repository is designed to be reproducible once the external raw datasets are available locally. The final derived artifacts are generated from the pipeline rather than stored in GitHub.

In practice, this means:

- GitHub stores the code, notebooks, and lightweight versioned inputs;
- the raw mobility, ticket, and GTFS assets live outside GitHub because of size restrictions;
- derived datasets are created locally by running the preprocessing and build notebooks.

## Recommended notebook order

### Foundation and preprocessing
1. `notebooks/01_eda_weather.ipynb`
2. `notebooks/02_eda_ticket.ipynb`
3. `notebooks/03_eda_mobility.ipynb`
4. `notebooks/00_build_preprocessed_datasets.ipynb`

### Integrated analysis
5. `notebooks/04_build_analytic_base.ipynb`
6. `notebooks/05_analysis_demand_weather.ipynb`
7. `notebooks/06_analysis_service_vs_weather_gtfs.ipynb`
8. `notebooks/07_analysis_route_resilience.ipynb`
9. `notebooks/08_analysis_joint_demand_service.ipynb`
10. `notebooks/09_storytelling_outputs.ipynb`

### Main reading path for evaluators

For a guided reading experience, start with:

1. `notebooks/10_final_narrative_index.ipynb`
2. `notebooks/09_storytelling_outputs.ipynb`
3. `notebooks/05_analysis_demand_weather.ipynb`
4. `notebooks/06_analysis_service_vs_weather_gtfs.ipynb`
5. `notebooks/07_analysis_route_resilience.ipynb`
6. `notebooks/08_analysis_joint_demand_service.ipynb`

Use `00` and `04` as technical support notebooks for preprocessing and analytic-base validation.

## Environment setup

Install the analysis dependencies:

```bash
pip install -r requirements-analysis.txt
```

Then ensure the raw data is available locally in the expected paths before executing the final notebooks.

## Final delivery orientation

The final analytical package is organized around one main narrative notebook:

- `notebooks/10_final_narrative_index.ipynb`

This notebook acts as the official entry point for the project, summarizing the problem, explaining the notebook pipeline, documenting data access, and directing the evaluator to the detailed evidence in the executed notebooks.
