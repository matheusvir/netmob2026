from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load ticket transactions, but only transaction timestamp is needed for demand counts.
base = Path('/home/jgluiggi/programming/netmob/data/ticket_data')
parts = []
for file in sorted(base.glob('*.csv')):
    df = pd.read_csv(file, usecols=['transaction_date'])
    dt = pd.to_datetime(df['transaction_date'], utc=True).dt.tz_convert('America/Sao_Paulo')
    df['date'] = dt.dt.normalize()
    df['hour'] = dt.dt.hour
    df['dow'] = dt.dt.dayofweek
    parts.append(df)
tickets = pd.concat(parts, ignore_index=True)

daily = tickets.groupby('date', as_index=False).size().rename(columns={'size': 'ticket_rows'})
daily['dow'] = daily['date'].dt.dayofweek
hourly = tickets.groupby(['date', 'hour'], as_index=False).size().rename(columns={'size': 'ticket_rows'})
hourly['dow'] = hourly['date'].dt.dayofweek

# Load weather and align to BRT dates.
weather = pd.read_csv('/home/jgluiggi/programming/netmob2026/data/meteorological_data.csv')
weather = weather.rename(columns={
    'Timestamp (UTC)': 'timestamp_utc',
    'Rain (mm)': 'rain_mm',
    'Wind Gust (m/s)': 'wind_gust',
    'Wind Speed (m/s)': 'wind_speed',
})
weather['timestamp_utc'] = pd.to_datetime(weather['timestamp_utc'], utc=True)
brt = weather['timestamp_utc'].dt.tz_convert('America/Sao_Paulo')
weather['date'] = brt.dt.normalize()
weather['hour'] = brt.dt.hour
march = weather[(weather['date'] >= pd.Timestamp('2026-03-01', tz='America/Sao_Paulo')) & (weather['date'] < pd.Timestamp('2026-04-01', tz='America/Sao_Paulo'))].copy()

# Daily weather categories.
daily_weather = march.groupby('date', as_index=False).agg(
    rain_mm=('rain_mm', 'sum'),
    wind_gust_max=('wind_gust', 'max'),
)
daily_weather['weather_cat'] = pd.cut(
    daily_weather['rain_mm'],
    bins=[-np.inf, 1, 10, 25, np.inf],
    labels=['Clear', 'Light Rain', 'Moderate Rain', 'Heavy Rain / Storm'],
    right=False,
)

daily_merged = daily.merge(daily_weather, on='date', how='left')

print('=== Coverage ===')
print('ticket days:', daily_merged['date'].nunique())
print('weather days:', daily_weather['date'].nunique())
print('hourly weather rows in March:', len(march))
print('days with heavy rain / storm:', int((daily_weather['rain_mm'] >= 25).sum()))
print('hourly observations with rain >= 10 mm:', int((march['rain_mm'] >= 10).sum()))
print()

print('=== Daily demand by weather category ===')
summary = daily_merged.groupby('weather_cat', observed=False)['ticket_rows'].agg(['count', 'mean', 'median', 'min', 'max'])
print(summary.reindex(['Clear', 'Light Rain', 'Moderate Rain', 'Heavy Rain / Storm']).to_string())
print()

print('=== Weather days with rain >= 10 mm ===')
print(daily_weather.loc[daily_weather['rain_mm'] >= 10, ['date', 'rain_mm', 'wind_gust_max']].to_string(index=False))
print()

print('=== Ticket days sorted by date ===')
print(daily_merged[['date', 'ticket_rows', 'rain_mm', 'weather_cat']].sort_values('date').to_string(index=False))
print()

# Simple Poisson model at daily grain.
model_daily = smf.glm('ticket_rows ~ rain_mm + C(dow)', data=daily_merged, family=sm.families.Poisson()).fit(cov_type='HC1')
print('=== Daily Poisson model ===')
print(model_daily.summary().tables[1].as_text())
print()

# Hourly Poisson model.
hourly_weather = march[['date', 'hour', 'rain_mm']].copy()
hourly_weather['rain_10mm'] = (hourly_weather['rain_mm'] >= 10).astype(int)
hourly_merged = hourly.merge(hourly_weather, on=['date', 'hour'], how='left')
model_hourly = smf.glm('ticket_rows ~ rain_mm + C(hour) + C(dow)', data=hourly_merged, family=sm.families.Poisson()).fit(cov_type='HC1')
print('=== Hourly Poisson model ===')
print(model_hourly.summary().tables[1].as_text())
print()
print('rain_mm coefficient (hourly):', model_hourly.params['rain_mm'])
print('rain_mm p-value (hourly):', model_hourly.pvalues['rain_mm'])
print('rain_10mm hours:', int(hourly_weather['rain_10mm'].sum()))
print('hours with rain >= 10 mm:')
print(hourly_weather.loc[hourly_weather['rain_10mm'] == 1, ['date', 'hour', 'rain_mm']].to_string(index=False))
