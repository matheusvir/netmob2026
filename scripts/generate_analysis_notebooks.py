from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14",
            },
        },
    )
    path = NOTEBOOKS_DIR / name
    nbf.write(notebook, path)
    print(f"Wrote {path}")


COMMON_SETUP = code(
    """
    from pathlib import Path
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', '{:,.4f}'.format)
    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams.update({'figure.figsize': (12, 5), 'figure.dpi': 120})


    def locate_project_root() -> Path:
        start = Path.cwd().resolve()
        for base in [start, *start.parents]:
            if (base / 'data' / 'derived').exists():
                return base
        raise FileNotFoundError('Could not locate the repo root from the current working directory.')


    PROJECT_ROOT = locate_project_root()
    DERIVED = PROJECT_ROOT / 'data' / 'derived'
    FIGURES = DERIVED / 'figures'
    FIGURES.mkdir(parents=True, exist_ok=True)
    WEATHER_ORDER = ['Clear', 'Light Rain', 'Moderate Rain', 'Heavy Rain / Storm']


    def savefig(name: str) -> Path:
        path = FIGURES / name
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print(f'Saved figure: {path}')
        return path
    """
)


NOTEBOOK_04 = [
    md(
        """
        # 04 · Build analytic base

        Este notebook audita a base analítica integrada construída para a fase final da análise.
        Ele verifica cobertura horária do clima, flags de qualidade da mobilidade, reconciliação
        de rotas e a janela integrada de 19 dias usada nos modelos que combinam ticket + clima + mobilidade.
        """
    ),
    COMMON_SETUP,
    code(
        """
        weather = pd.read_parquet(DERIVED / 'weather_hourly.parquet')
        ticket = pd.read_parquet(DERIVED / 'ticket_hourly_route_profile.parquet')
        mobility = pd.read_parquet(DERIVED / 'mobility_hourly_route_direction.parquet')
        integrated = pd.read_parquet(DERIVED / 'integrated_route_hour.parquet')
        quality = pd.read_csv(DERIVED / 'mobility_day_quality_flags.csv', parse_dates=['date'])
        crosswalk = pd.read_csv(DERIVED / 'route_crosswalk.csv')
        trip_coverage = pd.read_csv(DERIVED / 'trip_base_coverage.csv')

        print('=== Deliverables ===')
        print('weather_hourly rows:', len(weather))
        print('ticket_hourly_route_profile rows:', len(ticket))
        print('mobility_hourly_route_direction rows:', len(mobility))
        print('integrated_route_hour rows:', len(integrated))
        print()

        print('=== Weather hourly completeness ===')
        print('hour keys duplicated:', int(weather[['date', 'hour']].duplicated().sum()))
        print('missing observation rows:', int(weather['weather_observation_missing'].sum()))
        print(weather.loc[weather['weather_observation_missing'], ['date', 'hour']].to_string(index=False))
        print()

        print('=== Ticket coverage ===')
        print('ticket days:', pd.to_datetime(ticket['date']).dt.date.nunique())
        print('profiles:', sorted(ticket['card_label'].dropna().astype(str).unique().tolist()))
        print('boardings sum:', int(ticket['boardings'].sum()))
        """
    ),
    code(
        """
        print('=== Mobility quality flags ===')
        print(quality[['date', 'row_count', 'row_count_ratio', 'is_partial_day', 'partial_day_reason']].to_string(index=False))
        print()

        print('=== Route crosswalk resolution ===')
        print(crosswalk['route_crosswalk_confidence'].value_counts(dropna=False).to_string())
        print()
        print('Unresolved routes excluded from integrated models:')
        print(crosswalk.loc[crosswalk['manual_review'], ['route_norm', 'rationale']].drop_duplicates().to_string(index=False))
        print()

        print('=== GTFS trip-base coverage ===')
        matched = int(trip_coverage['in_gtfs'].sum())
        total = int(len(trip_coverage))
        print(f'matched trip bases: {matched}/{total}')
        print('unmatched trip bases:', sorted(trip_coverage.loc[~trip_coverage['in_gtfs'], 'trip_base'].tolist()))
        """
    ),
    code(
        """
        print('=== Integrated 19-day window ===')
        integrated_dates = pd.to_datetime(integrated['date'])
        print('unique integrated days:', integrated_dates.dt.date.nunique())
        print('min date:', integrated_dates.min())
        print('max date:', integrated_dates.max())
        print('route_crosswalk_confidence present:', 'route_crosswalk_confidence' in integrated.columns)
        print('coverage_flag present:', 'coverage_flag' in integrated.columns)
        print()

        print('Integrated confidence mix:')
        print(integrated['route_crosswalk_confidence'].value_counts(dropna=False).to_string())
        print()

        print('Rows with mobility metrics present:', int(integrated['observed_trip_count'].notna().sum()))
        print('Rows without mobility metrics present:', int(integrated['observed_trip_count'].isna().sum()))
        print()

        print('Integrated sample:')
        print(integrated.head(12).to_string(index=False))
        """
    ),
]


NOTEBOOK_05 = [
    md(
        """
        # 05 · Demand weather analysis

        Este notebook fecha **Q1** e **Q2** do plano:

        - **Q1:** como a demanda reage à chuva em março/2026
        - **Q2:** como essa resposta varia entre perfis de passageiro

        **Janela usada aqui:** a janela completa de **31 dias** de ticket + clima.
        A base inclui um aviso honesto: o clima em horário local tem **3 horas faltantes no fim de 31/03**,
        então qualquer leitura sobre o último dia deve ser interpretada com essa ressalva.
        """
    ),
    COMMON_SETUP,
    code(
        """
        weather = pd.read_parquet(DERIVED / 'weather_hourly.parquet')
        ticket = pd.read_parquet(DERIVED / 'ticket_hourly_route_profile.parquet')

        weather_clean = weather.loc[~weather['weather_observation_missing']].copy()
        weather_clean['weather_cat'] = pd.Categorical(weather_clean['weather_cat'], categories=WEATHER_ORDER, ordered=True)

        route_hourly = (
            ticket.groupby(['date', 'hour', 'route_norm', 'is_weekend', 'day_of_week'], observed=True, as_index=False)
            .agg(boardings=('boardings', 'sum'))
            .merge(weather_clean[['date', 'hour', 'rain_mm', 'rain_3h', 'rain_6h', 'weather_cat', 'wind_gust', 'temp_c']], on=['date', 'hour'], how='inner')
        )
        profile_hourly = (
            ticket.groupby(['date', 'hour', 'card_label', 'is_weekend', 'day_of_week'], observed=True, as_index=False)
            .agg(boardings=('boardings', 'sum'))
            .merge(weather_clean[['date', 'hour', 'rain_mm', 'weather_cat']], on=['date', 'hour'], how='inner')
        )
        profile_hourly['weather_cat'] = pd.Categorical(profile_hourly['weather_cat'], categories=WEATHER_ORDER, ordered=True)

        daily_ticket = ticket.groupby('date', as_index=False).agg(boardings=('boardings', 'sum'))
        daily_weather = (
            weather_clean.groupby('date', as_index=False)
            .agg(
                rain_mm=('rain_mm', 'sum'),
                wind_gust=('wind_gust', 'max'),
                temp_c=('temp_c', 'mean'),
                observed_hours=('hour', 'size'),
            )
        )
        daily_weather['weather_cat'] = pd.Categorical(
            pd.cut(daily_weather['rain_mm'], bins=[-np.inf, 1, 10, 25, np.inf], labels=WEATHER_ORDER, right=False),
            categories=WEATHER_ORDER,
            ordered=True,
        )
        daily_weather['is_weekend'] = pd.to_datetime(daily_weather['date']).dt.dayofweek >= 5
        daily_weather['day_of_week'] = pd.to_datetime(daily_weather['date']).dt.dayofweek
        daily = daily_ticket.merge(daily_weather, on='date', how='inner').sort_values('date').reset_index(drop=True)

        print('=== Sample composition ===')
        print('ticket days:', pd.to_datetime(ticket['date']).dt.date.nunique())
        print('weather days:', pd.to_datetime(weather_clean['date']).dt.date.nunique())
        print('route-hour rows:', len(route_hourly))
        print('profile-hour rows:', len(profile_hourly))
        print('profiles:', sorted(profile_hourly['card_label'].dropna().astype(str).unique().tolist()))
        print()

        print('=== Weather coverage by day ===')
        print(daily[['date', 'observed_hours', 'rain_mm', 'weather_cat']].tail(5).to_string(index=False))
        print()

        print('=== Daily weather category counts ===')
        print(daily['weather_cat'].value_counts(dropna=False).reindex(WEATHER_ORDER).to_string())
        print('Heavy Rain / Storm days:', int((daily['weather_cat'] == 'Heavy Rain / Storm').sum()))
        print('Hourly observations with rain >= 10 mm:', int((weather_clean['rain_mm'] >= 10).sum()))
        """
    ),
    code(
        """
        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax2 = ax1.twinx()
        ax1.bar(pd.to_datetime(daily['date']), daily['boardings'], color='#4C78A8', alpha=0.85)
        ax2.plot(pd.to_datetime(daily['date']), daily['rain_mm'], color='#D62728', marker='o', linewidth=2)
        ax1.set_title('Q1 · Daily demand and rainfall in March/2026')
        ax1.set_ylabel('Boardings')
        ax2.set_ylabel('Daily rainfall (mm)')
        ax1.tick_params(axis='x', rotation=45)
        savefig('q1_daily_demand_vs_rain.png')
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=route_hourly, x='weather_cat', y='boardings', order=WEATHER_ORDER, ax=ax)
        ax.set_title('Q1 · Route-hour demand by weather category')
        ax.set_xlabel('Weather category')
        ax.set_ylabel('Boardings per route-hour')
        ax.tick_params(axis='x', rotation=15)
        savefig('q1_route_hour_boxplot.png')
        plt.show()
        """
    ),
    code(
        """
        route_hourly = route_hourly.copy()
        route_hourly['weather_cat'] = pd.Categorical(route_hourly['weather_cat'], categories=WEATHER_ORDER, ordered=True)

        q1_model_all = smf.glm(
            'boardings ~ rain_mm + C(hour) + C(day_of_week)',
            data=route_hourly,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')
        q1_model_weekday = smf.glm(
            'boardings ~ rain_mm + C(hour) + C(day_of_week)',
            data=route_hourly.loc[~route_hourly['is_weekend']].copy(),
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')

        daily_model = smf.glm(
            'boardings ~ rain_mm + C(day_of_week)',
            data=daily,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')

        rng = np.random.default_rng(20260702)
        boot_coefs = []
        for _ in range(150):
            sample = daily.sample(n=len(daily), replace=True, random_state=int(rng.integers(0, 1_000_000)))
            try:
                boot_model = smf.glm(
                    'boardings ~ rain_mm + C(day_of_week)',
                    data=sample,
                    family=sm.families.Poisson(),
                ).fit()
                boot_coefs.append(float(boot_model.params['rain_mm']))
            except Exception:
                continue
        boot_ci = np.quantile(boot_coefs, [0.025, 0.975]) if boot_coefs else [np.nan, np.nan]

        q1_effects = pd.DataFrame(
            {
                'sample': ['all route-hours', 'weekday route-hours'],
                'rain_coef': [q1_model_all.params['rain_mm'], q1_model_weekday.params['rain_mm']],
                'rain_pvalue': [q1_model_all.pvalues['rain_mm'], q1_model_weekday.pvalues['rain_mm']],
                'pct_effect_per_mm': [100 * (np.exp(q1_model_all.params['rain_mm']) - 1), 100 * (np.exp(q1_model_weekday.params['rain_mm']) - 1)],
            }
        )
        print('=== Q1 Poisson models ===')
        print(q1_effects.to_string(index=False))
        print()
        print('Daily bootstrap 95% CI for rain_mm coefficient:', boot_ci)
        print('Daily model coefficient table:')
        print(daily_model.summary().tables[1].as_text())

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=q1_effects, x='sample', y='pct_effect_per_mm', ax=ax)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title('Q1 · Marginal effect of 1 mm of rain on demand')
        ax.set_ylabel('Percent effect (%)')
        ax.set_xlabel('')
        savefig('q1_rain_effect_models.png')
        plt.show()
        """
    ),
    code(
        """
        keep_profiles = ['Standard', 'Student', 'Labor', 'Senior', 'Cash']
        profile_q2 = profile_hourly.loc[profile_hourly['card_label'].isin(keep_profiles)].copy()

        q2_model = smf.glm(
            'boardings ~ rain_mm * C(card_label) + C(hour) + C(day_of_week)',
            data=profile_q2,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')

        interaction_rows = []
        for label in keep_profiles:
            if label == 'Cash':
                term = 'rain_mm'
                interaction_rows.append({'profile': label, 'term': term, 'coef': q2_model.params.get(term, np.nan), 'pvalue': q2_model.pvalues.get(term, np.nan)})
                continue
            term = f'rain_mm:C(card_label)[T.{label}]'
            interaction_rows.append({'profile': label, 'term': term, 'coef': q2_model.params.get(term, 0.0), 'pvalue': q2_model.pvalues.get(term, np.nan)})
        interaction_df = pd.DataFrame(interaction_rows)
        print('=== Q2 interaction coefficients ===')
        print(interaction_df.to_string(index=False))
        print()

        heatmap = (
            profile_q2.groupby(['card_label', 'weather_cat'], observed=True)['boardings']
            .mean()
            .unstack(fill_value=np.nan)
            .reindex(index=keep_profiles, columns=WEATHER_ORDER)
        )
        normalized_heatmap = heatmap.div(heatmap['Clear'], axis=0) - 1
        print('=== Q2 normalized profile x weather table ===')
        print((normalized_heatmap * 100).round(2).to_string())

        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.heatmap(normalized_heatmap * 100, annot=True, fmt='.1f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Q2 · Relative change vs clear weather by passenger profile (%)')
        ax.set_xlabel('Weather category')
        ax.set_ylabel('Passenger profile')
        savefig('q2_profile_weather_heatmap.png')
        plt.show()

        profile_q2['rain_flag'] = np.where(profile_q2['rain_mm'].fillna(0) > 0, 'Rain', 'Clear')
        curve = (
            profile_q2.loc[profile_q2['rain_flag'].isin(['Clear', 'Rain'])]
            .groupby(['card_label', 'hour', 'rain_flag'], observed=True, as_index=False)
            .agg(boardings=('boardings', 'mean'))
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, label in zip(axes, ['Standard', 'Student']):
            sns.lineplot(data=curve.loc[curve['card_label'] == label], x='hour', y='boardings', hue='rain_flag', marker='o', ax=ax)
            ax.set_title(f'Q2 · Hourly curve — {label}')
            ax.set_xlabel('Hour of day')
            ax.set_ylabel('Mean boardings per hour')
        savefig('q2_profile_hourly_curves.png')
        plt.show()
        """
    ),
    md(
        """
        ## Conclusão

        - **Q1:** a demanda responde à chuva dentro da janela completa de ticket + clima, e o notebook deixa explícita a comparação entre **toda a amostra** e a sensibilidade **apenas em dias úteis**.
        - **Q2:** a heterogeneidade por perfil agora está operacionalizada com `card_label` legível (`Standard`, `Student`, `Labor`, `Senior`, `Cash`), com modelo de interação e visualizações dedicadas.
        - **Limite importante:** março/2026 **não contém observações de `Heavy Rain / Storm`**, então a conclusão deve permanecer no escopo de **tempo firme vs chuva leve/moderada**.
        """
    ),
]


NOTEBOOK_06 = [
    md(
        """
        # 06 · Service vs weather (GTFS-enhanced)

        Este notebook responde **Q3** do plano: se a chuva está associada a piora operacional em
        `headway_p50`, `speed_p50`, `service_gap_index` e `observed_vs_scheduled_trip_ratio`.

        **Janela usada aqui:** a janela integrada de **19 dias** com ticket + clima + mobilidade.
        """
    ),
    COMMON_SETUP,
    code(
        """
        integrated = pd.read_parquet(DERIVED / 'integrated_route_hour.parquet')
        quality = pd.read_csv(DERIVED / 'mobility_day_quality_flags.csv', parse_dates=['date'])

        service_all = integrated.dropna(subset=['observed_trip_count']).copy()
        service_all['weather_bucket'] = pd.Categorical(
            np.where(
                service_all['rain_mm'].fillna(0) >= 10,
                'Adverse (>=10 mm)',
                np.where(service_all['rain_mm'].fillna(0) > 0, 'Rain (<10 mm)', 'Clear'),
            ),
            categories=['Clear', 'Rain (<10 mm)', 'Adverse (>=10 mm)'],
            ordered=True,
        )
        service_clean = service_all.loc[service_all['coverage_flag'].fillna('ok') != 'partial_day'].copy()

        print('=== Integrated service sample ===')
        print('unique integrated days:', pd.to_datetime(service_all['date']).dt.date.nunique())
        print('service rows (all):', len(service_all))
        print('service rows (excluding partial-day flags):', len(service_clean))
        print('flagged mobility days:', quality.loc[quality['is_partial_day'], 'date'].dt.strftime('%Y-%m-%d').tolist())
        print()

        summary = service_clean.groupby('weather_bucket', observed=False)[['headway_p50', 'speed_p50', 'service_gap_index', 'observed_vs_scheduled_trip_ratio']].agg(['count', 'mean', 'median'])
        print('=== Weather-bucket summary on clean sample ===')
        print(summary.to_string())
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        sns.boxplot(data=service_clean, x='weather_bucket', y='headway_p50', ax=axes[0])
        axes[0].set_title('Headway p50 by weather bucket')
        sns.boxplot(data=service_clean, x='weather_bucket', y='speed_p50', ax=axes[1])
        axes[1].set_title('Speed p50 by weather bucket')
        sns.boxplot(data=service_clean, x='weather_bucket', y='service_gap_index', ax=axes[2])
        axes[2].set_title('Service gap index by weather bucket')
        for ax in axes:
            ax.tick_params(axis='x', rotation=15)
        savefig('q3_service_boxplots.png')
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(data=service_clean, x='weather_bucket', y='observed_vs_scheduled_trip_ratio', ax=ax)
        ax.set_title('Observed / scheduled trip ratio by weather bucket')
        ax.tick_params(axis='x', rotation=15)
        savefig('q3_schedule_ratio_by_weather.png')
        plt.show()
        """
    ),
    code(
        """
        def fit_service_models(frame: pd.DataFrame, label: str) -> pd.DataFrame:
            output = []
            specs = {
                'headway_p50': frame.dropna(subset=['headway_p50']).copy(),
                'speed_p50': frame.dropna(subset=['speed_p50']).copy(),
                'service_gap_index': frame.dropna(subset=['service_gap_index']).copy(),
            }
            for metric, df_metric in specs.items():
                model = smf.ols(
                    f'{metric} ~ rain_mm + C(route_norm) + C(hour) + C(day_of_week)',
                    data=df_metric,
                ).fit(cov_type='HC1')
                output.append(
                    {
                        'sample': label,
                        'metric': metric,
                        'rain_coef': model.params['rain_mm'],
                        'rain_pvalue': model.pvalues['rain_mm'],
                        'nobs': int(model.nobs),
                    }
                )
            return pd.DataFrame(output)


        sensitivity = pd.concat(
            [
                fit_service_models(service_all, 'all mobility days'),
                fit_service_models(service_clean, 'excluding partial days'),
            ],
            ignore_index=True,
        )
        print('=== Sensitivity comparison ===')
        print(sensitivity.to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(data=sensitivity, x='metric', y='rain_coef', hue='sample', ax=ax)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title('Q3 · Rain coefficient with vs without partial-day mobility flags')
        ax.set_ylabel('Coefficient on rain_mm')
        ax.set_xlabel('')
        savefig('q3_sensitivity_flagged_days.png')
        plt.show()
        """
    ),
    md(
        """
        ## Leitura preliminar

        - O notebook reporta a amostra operacional na **janela integrada de 19 dias**.
        - A sensibilidade **com vs sem** dias parciais foi explicitamente rodada, em vez de apenas filtrar silenciosamente.
        - As conclusões de Q3 devem ser formuladas para **chuva leve/moderada**, porque não há base observacional para tempestades severas nessa janela.
        """
    ),
]


NOTEBOOK_07 = [
    md(
        """
        # 07 · Route resilience

        Este notebook responde **Q4**: quais rotas são mais resilientes ou mais vulneráveis quando a chuva aparece,
        combinando variação de demanda, piora de headway, perda de velocidade e gap entre serviço observado e programado.

        **Janela usada aqui:** a janela integrada de **19 dias**.
        """
    ),
    COMMON_SETUP,
    code(
        """
        from scipy.stats import spearmanr
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        integrated = pd.read_parquet(DERIVED / 'integrated_route_hour.parquet')
        base = integrated.dropna(subset=['observed_trip_count']).copy()
        base = base.loc[base['coverage_flag'].fillna('ok') != 'partial_day'].copy()
        base['rain_flag'] = np.where(base['rain_mm'].fillna(0) > 0, 'Rain', 'Clear')
        print('=== Sample ===')
        print('rows:', len(base))
        print('unique routes:', base['route_norm'].nunique())
        print('unique days:', pd.to_datetime(base['date']).dt.date.nunique())
        """
    ),
    code(
        """
        route_weather = (
            base.groupby(['route_norm', 'operator', 'rain_flag'], observed=True)
            .agg(
                boardings=('boardings', 'mean'),
                headway_p50=('headway_p50', 'mean'),
                speed_p50=('speed_p50', 'mean'),
                service_gap_index=('service_gap_index', 'mean'),
                observed_vs_scheduled_trip_ratio=('observed_vs_scheduled_trip_ratio', 'mean'),
                route_crosswalk_confidence=('route_crosswalk_confidence', 'first'),
            )
            .reset_index()
        )

        clear = route_weather.loc[route_weather['rain_flag'] == 'Clear'].copy()
        rain = route_weather.loc[route_weather['rain_flag'] == 'Rain'].copy()
        resilience = clear.merge(rain, on=['route_norm', 'operator', 'route_crosswalk_confidence'], suffixes=('_clear', '_rain'))
        resilience['demand_delta_pct'] = 100 * (resilience['boardings_rain'] / resilience['boardings_clear'] - 1)
        resilience['headway_delta_pct'] = 100 * (resilience['headway_p50_rain'] / resilience['headway_p50_clear'] - 1)
        resilience['speed_delta_pct'] = 100 * (resilience['speed_p50_rain'] / resilience['speed_p50_clear'] - 1)
        resilience['gap_delta'] = resilience['service_gap_index_rain'] - resilience['service_gap_index_clear']
        resilience['schedule_ratio_delta'] = resilience['observed_vs_scheduled_trip_ratio_rain'] - resilience['observed_vs_scheduled_trip_ratio_clear']

        feature_cols = ['demand_delta_pct', 'headway_delta_pct', 'speed_delta_pct', 'gap_delta', 'schedule_ratio_delta']
        scored = resilience.dropna(subset=feature_cols).copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(scored[feature_cols])
        scored['resilience_index'] = (
            -scaled[:, 0]
            -scaled[:, 1]
            +scaled[:, 2]
            -scaled[:, 3]
            +scaled[:, 4]
        ) / 5
        scored['alternative_index'] = (
            -scored['demand_delta_pct'].rank(pct=True)
            -scored['headway_delta_pct'].rank(pct=True)
            +scored['speed_delta_pct'].rank(pct=True)
            -scored['gap_delta'].rank(pct=True)
            +scored['schedule_ratio_delta'].rank(pct=True)
        ) / 5

        rank_corr = spearmanr(scored['resilience_index'], scored['alternative_index']).statistic
        print('=== Route-level resilience panel ===')
        print('routes with clear+rain coverage:', len(scored))
        print('Spearman correlation between main and alternative index:', rank_corr)
        print()
        print(scored[['route_norm', 'operator', 'resilience_index', 'alternative_index']].sort_values('resilience_index', ascending=False).head(10).to_string(index=False))
        """
    ),
    code(
        """
        k = min(3, len(scored)) if len(scored) else 1
        if len(scored):
            clusters = KMeans(n_clusters=max(k, 1), n_init=20, random_state=20260702).fit_predict(scaled)
            scored['cluster'] = clusters.astype(str)
        else:
            scored['cluster'] = []

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        best = scored.sort_values('resilience_index', ascending=False).head(10)
        worst = scored.sort_values('resilience_index', ascending=True).head(10)
        sns.barplot(data=best, x='route_norm', y='resilience_index', hue='operator', ax=axes[0])
        axes[0].set_title('Top resilient routes')
        axes[0].tick_params(axis='x', rotation=45)
        sns.barplot(data=worst, x='route_norm', y='resilience_index', hue='operator', ax=axes[1])
        axes[1].set_title('Top vulnerable routes')
        axes[1].tick_params(axis='x', rotation=45)
        savefig('q4_resilience_rankings.png')
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=scored, x='demand_delta_pct', y='headway_delta_pct', hue='cluster', style='operator', s=90, ax=ax)
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_title('Q4 · Route clusters in resilience feature space')
        ax.set_xlabel('Demand delta (%)')
        ax.set_ylabel('Headway delta (%)')
        savefig('q4_route_clusters.png')
        plt.show()
        """
    ),
    md(
        """
        ## Leitura preliminar

        - A rota só entra no ranking se tiver observações tanto em **tempo firme** quanto em **hora chuvosa**.
        - O índice principal foi comparado com uma definição alternativa; a estabilidade é reportada pela correlação de Spearman.
        - As rotas permanecem marcadas pelo `operator` e pelo `route_crosswalk_confidence`, para evitar superinterpretação de matches manuais.
        """
    ),
]


NOTEBOOK_08 = [
    md(
        """
        # 08 · Joint demand × service analysis

        Este notebook responde **Q5**: a piora de serviço ajuda a explicar a queda de demanda em horas chuvosas?

        **Janela usada aqui:** a janela integrada de **19 dias**.
        **Regra de interpretação:** esta análise continua sendo **observacional / exploratória**, sem reivindicação causal forte.
        """
    ),
    COMMON_SETUP,
    code(
        """
        integrated = pd.read_parquet(DERIVED / 'integrated_route_hour.parquet')
        model_df = integrated.loc[integrated['coverage_flag'].fillna('ok') != 'partial_day'].copy()
        model_df = model_df.dropna(
            subset=['boardings', 'rain_mm', 'headway_p50', 'speed_p50', 'service_gap_index', 'hour', 'day_of_week', 'route_norm']
        ).copy()

        print('=== Modeling sample ===')
        print('rows:', len(model_df))
        print('routes:', model_df['route_norm'].nunique())
        print('days:', pd.to_datetime(model_df['date']).dt.date.nunique())
        print(model_df[['date', 'hour', 'route_norm', 'boardings', 'rain_mm', 'headway_p50', 'speed_p50', 'service_gap_index']].head(10).to_string(index=False))
        """
    ),
    code(
        """
        base_model = smf.glm(
            'boardings ~ rain_mm + C(route_norm) + C(hour) + C(day_of_week)',
            data=model_df,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')

        service_model = smf.glm(
            'boardings ~ rain_mm + headway_p50 + speed_p50 + service_gap_index + C(route_norm) + C(hour) + C(day_of_week)',
            data=model_df,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')

        interaction_model = smf.glm(
            'boardings ~ rain_mm + headway_p50 + speed_p50 + service_gap_index + rain_mm:headway_p50 + rain_mm:speed_p50 + rain_mm:service_gap_index + C(route_norm) + C(hour) + C(day_of_week)',
            data=model_df,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')

        comparison = pd.DataFrame(
            {
                'model': ['base', 'service', 'interaction'],
                'rain_coef': [base_model.params['rain_mm'], service_model.params['rain_mm'], interaction_model.params['rain_mm']],
                'rain_pvalue': [base_model.pvalues['rain_mm'], service_model.pvalues['rain_mm'], interaction_model.pvalues['rain_mm']],
                'aic': [base_model.aic, service_model.aic, interaction_model.aic],
                'pct_effect_per_mm': [100 * (np.exp(base_model.params['rain_mm']) - 1), 100 * (np.exp(service_model.params['rain_mm']) - 1), 100 * (np.exp(interaction_model.params['rain_mm']) - 1)],
            }
        )
        print('=== Nested model comparison ===')
        print(comparison.to_string(index=False))
        print()

        interaction_terms = pd.DataFrame(
            {
                'term': ['rain_mm:headway_p50', 'rain_mm:speed_p50', 'rain_mm:service_gap_index'],
                'coef': [interaction_model.params.get('rain_mm:headway_p50', np.nan), interaction_model.params.get('rain_mm:speed_p50', np.nan), interaction_model.params.get('rain_mm:service_gap_index', np.nan)],
                'pvalue': [interaction_model.pvalues.get('rain_mm:headway_p50', np.nan), interaction_model.pvalues.get('rain_mm:speed_p50', np.nan), interaction_model.pvalues.get('rain_mm:service_gap_index', np.nan)],
            }
        )
        print('=== Interaction terms ===')
        print(interaction_terms.to_string(index=False))
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        sns.barplot(data=comparison, x='model', y='pct_effect_per_mm', ax=axes[0])
        axes[0].axhline(0, color='black', linewidth=1)
        axes[0].set_title('Q5 · Rain effect across nested models')
        axes[0].set_ylabel('Percent effect of 1 mm rain')

        sns.barplot(data=interaction_terms, x='term', y='coef', ax=axes[1])
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_title('Q5 · Interaction coefficients')
        axes[1].tick_params(axis='x', rotation=20)
        savefig('q5_nested_model_comparison.png')
        plt.show()
        """
    ),
    md(
        """
        ## Leitura preliminar

        - Se `rain_mm` perde magnitude quando as variáveis operacionais entram no modelo, parte da queda de demanda é compatível com **mediação via piora operacional**.
        - Se um termo de interação fica significativo, isso é o indício mais forte para a leitura de **amplificação**.
        - Mesmo assim, o resultado precisa ser comunicado como **evidência associativa** dentro da janela integrada de 19 dias.
        """
    ),
]


NOTEBOOK_09 = [
    md(
        """
        # 09 · Storytelling outputs

        Este notebook consolida saídas visuais finais (`final_*`) para a apresentação.
        Ele resume Q1–Q5 e explicita as limitações metodológicas que não podem ser omitidas.
        """
    ),
    COMMON_SETUP,
    code(
        """
        weather = pd.read_parquet(DERIVED / 'weather_hourly.parquet')
        ticket = pd.read_parquet(DERIVED / 'ticket_hourly_route_profile.parquet')
        integrated = pd.read_parquet(DERIVED / 'integrated_route_hour.parquet')
        quality = pd.read_csv(DERIVED / 'mobility_day_quality_flags.csv', parse_dates=['date'])
        trip_coverage = pd.read_csv(DERIVED / 'trip_base_coverage.csv')
        crosswalk = pd.read_csv(DERIVED / 'route_crosswalk.csv')

        weather_clean = weather.loc[~weather['weather_observation_missing']].copy()
        daily_ticket = ticket.groupby('date', as_index=False).agg(boardings=('boardings', 'sum'))
        daily_weather = weather_clean.groupby('date', as_index=False).agg(rain_mm=('rain_mm', 'sum'))
        daily_weather['weather_cat'] = pd.Categorical(
            pd.cut(daily_weather['rain_mm'], bins=[-np.inf, 1, 10, 25, np.inf], labels=WEATHER_ORDER, right=False),
            categories=WEATHER_ORDER,
            ordered=True,
        )
        daily = daily_ticket.merge(daily_weather, on='date', how='inner')

        print('=== Final storytelling context ===')
        print('Ticket days:', pd.to_datetime(ticket['date']).dt.date.nunique())
        print('Integrated days:', pd.to_datetime(integrated['date']).dt.date.nunique())
        print('Flagged mobility days:', int(quality['is_partial_day'].sum()))
        print('Unmatched trip bases:', int((~trip_coverage['in_gtfs']).sum()))
        print('Unresolved ticket routes excluded:', int(crosswalk['manual_review'].sum()))
        """
    ),
    code(
        """
        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax2 = ax1.twinx()
        ax1.bar(pd.to_datetime(daily['date']), daily['boardings'], color='#4C78A8', alpha=0.85)
        ax2.plot(pd.to_datetime(daily['date']), daily['rain_mm'], color='#D62728', marker='o', linewidth=2)
        ax1.set_title('Final Q1 · Demand vs rainfall in March/2026')
        ax1.set_ylabel('Boardings')
        ax2.set_ylabel('Daily rainfall (mm)')
        ax1.tick_params(axis='x', rotation=45)
        savefig('final_q1_demand_weather.png')
        plt.show()

        profile_weather = (
            ticket.groupby(['card_label', 'date', 'hour'], observed=True, as_index=False)
            .agg(boardings=('boardings', 'sum'))
            .merge(weather_clean[['date', 'hour', 'weather_cat']], on=['date', 'hour'], how='inner')
        )
        heatmap = (
            profile_weather.groupby(['card_label', 'weather_cat'], observed=True)['boardings']
            .mean()
            .unstack(fill_value=np.nan)
            .reindex(columns=WEATHER_ORDER)
        )
        normalized = heatmap.div(heatmap['Clear'], axis=0) - 1
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.heatmap(normalized * 100, cmap='coolwarm', center=0, annot=True, fmt='.1f', ax=ax)
        ax.set_title('Final Q2 · Relative change vs clear weather by passenger profile (%)')
        savefig('final_q2_profile_heterogeneity.png')
        plt.show()
        """
    ),
    code(
        """
        service = integrated.dropna(subset=['observed_trip_count']).copy()
        service = service.loc[service['coverage_flag'].fillna('ok') != 'partial_day'].copy()
        service['weather_bucket'] = pd.Categorical(
            np.where(service['rain_mm'].fillna(0) >= 10, 'Adverse (>=10 mm)', np.where(service['rain_mm'].fillna(0) > 0, 'Rain (<10 mm)', 'Clear')),
            categories=['Clear', 'Rain (<10 mm)', 'Adverse (>=10 mm)'],
            ordered=True,
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(data=service, x='weather_bucket', y='service_gap_index', ax=ax)
        ax.set_title('Final Q3 · Service gap index under different weather buckets')
        ax.tick_params(axis='x', rotation=15)
        savefig('final_q3_service_degradation.png')
        plt.show()

        route_weather = (
            service.assign(rain_flag=np.where(service['rain_mm'].fillna(0) > 0, 'Rain', 'Clear'))
            .groupby(['route_norm', 'operator', 'rain_flag'], observed=True)
            .agg(boardings=('boardings', 'mean'), headway_p50=('headway_p50', 'mean'))
            .reset_index()
        )
        clear = route_weather.loc[route_weather['rain_flag'] == 'Clear'].copy()
        rain = route_weather.loc[route_weather['rain_flag'] == 'Rain'].copy()
        route_delta = clear.merge(rain, on=['route_norm', 'operator'], suffixes=('_clear', '_rain'))
        route_delta['resilience_score'] = -(route_delta['boardings_rain'] / route_delta['boardings_clear'] - 1) - (route_delta['headway_p50_rain'] / route_delta['headway_p50_clear'] - 1)
        top_vulnerable = route_delta.sort_values('resilience_score', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=top_vulnerable, x='route_norm', y='resilience_score', hue='operator', ax=ax)
        ax.set_title('Final Q4 · Most vulnerable routes in rainy conditions')
        ax.tick_params(axis='x', rotation=45)
        savefig('final_q4_route_resilience.png')
        plt.show()
        """
    ),
    code(
        """
        model_df = integrated.loc[integrated['coverage_flag'].fillna('ok') != 'partial_day'].copy()
        model_df = model_df.dropna(subset=['boardings', 'rain_mm', 'headway_p50', 'speed_p50', 'service_gap_index', 'hour', 'day_of_week', 'route_norm']).copy()
        base_model = smf.glm(
            'boardings ~ rain_mm + C(route_norm) + C(hour) + C(day_of_week)',
            data=model_df,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')
        service_model = smf.glm(
            'boardings ~ rain_mm + headway_p50 + speed_p50 + service_gap_index + C(route_norm) + C(hour) + C(day_of_week)',
            data=model_df,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')
        interaction_model = smf.glm(
            'boardings ~ rain_mm + headway_p50 + speed_p50 + service_gap_index + rain_mm:headway_p50 + rain_mm:speed_p50 + rain_mm:service_gap_index + C(route_norm) + C(hour) + C(day_of_week)',
            data=model_df,
            family=sm.families.Poisson(),
        ).fit(cov_type='HC1')
        nested = pd.DataFrame(
            {
                'model': ['base', 'service', 'interaction'],
                'rain_pct_effect': [100 * (np.exp(base_model.params['rain_mm']) - 1), 100 * (np.exp(service_model.params['rain_mm']) - 1), 100 * (np.exp(interaction_model.params['rain_mm']) - 1)],
                'aic': [base_model.aic, service_model.aic, interaction_model.aic],
            }
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=nested, x='model', y='rain_pct_effect', ax=ax)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title('Final Q5 · Rain effect across nested demand models')
        ax.set_ylabel('Percent effect of 1 mm rain')
        savefig('final_q5_joint_effects.png')
        plt.show()
        """
    ),
    code(
        """
        limitations = pd.DataFrame(
            {
                'limitation': [
                    'Q3–Q5 rely on the integrated 19-day window',
                    'Heavy Rain / Storm is absent from the observed month',
                    'Five mobility trip bases remain unmatched to GTFS',
                    'Five ticket routes remain unresolved and were excluded from integrated joins',
                    'Four mobility days were flagged as partial and tested via sensitivity analyses',
                ]
            }
        )
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.axis('off')
        table = ax.table(
            cellText=[[idx + 1, value] for idx, value in enumerate(limitations['limitation'])],
            colLabels=['#', 'Methodological limitation'],
            loc='center',
            cellLoc='left',
            colLoc='left',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        ax.set_title('Final limitations that must remain in the presentation')
        savefig('final_limitations_table.png')
        plt.show()
        print(limitations.to_string(index=False))
        """
    ),
    md(
        """
        ## Mensagens seguras para a apresentação

        1. **Q1:** há base suficiente para discutir demanda em tempo firme, chuva leve e chuva moderada.
        2. **Q2:** a resposta à chuva não precisa ser homogênea entre perfis de passageiro; a heterogeneidade foi testada explicitamente.
        3. **Q3–Q5:** os resultados operacionais e conjuntos devem ser comunicados como achados da **janela integrada de 19 dias**, não do mês inteiro.
        4. **Eventos severos:** como `Heavy Rain / Storm` não aparece nas observações, não se deve extrapolar para tempestades fortes.
        """
    ),
]


def main() -> None:
    write_notebook('04_build_analytic_base.ipynb', NOTEBOOK_04)
    write_notebook('05_analysis_demand_weather.ipynb', NOTEBOOK_05)
    write_notebook('06_analysis_service_vs_weather_gtfs.ipynb', NOTEBOOK_06)
    write_notebook('07_analysis_route_resilience.ipynb', NOTEBOOK_07)
    write_notebook('08_analysis_joint_demand_service.ipynb', NOTEBOOK_08)
    write_notebook('09_storytelling_outputs.ipynb', NOTEBOOK_09)


if __name__ == '__main__':
    main()
