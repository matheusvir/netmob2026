from pathlib import Path
import pandas as pd
import numpy as np

w = pd.read_csv('/home/jgluiggi/programming/netmob2026/data/meteorological_data.csv')
w = w.rename(columns={
    'Timestamp (UTC)': 'timestamp_utc',
    'Rain (mm)': 'rain_mm',
    'Wind Gust (m/s)': 'wind_gust',
    'Wind Speed (m/s)': 'wind_speed',
})
w['timestamp_utc'] = pd.to_datetime(w['timestamp_utc'], utc=True)
brt = w['timestamp_utc'].dt.tz_convert('America/Sao_Paulo')
w['date'] = brt.dt.normalize()
w['hour'] = brt.dt.hour
march = w[(w['date'] >= pd.Timestamp('2026-03-01', tz='America/Sao_Paulo')) & (w['date'] < pd.Timestamp('2026-04-01', tz='America/Sao_Paulo'))].copy()
print('march rows', len(march), 'dates', march['date'].nunique())
wd = march.groupby('date', as_index=False).agg(rain_mm=('rain_mm', 'sum'), wind_gust_max=('wind_gust', 'max'))
wd['weather_cat'] = pd.cut(
    wd['rain_mm'],
    bins=[-np.inf, 1, 10, 25, np.inf],
    labels=['Clear', 'Light Rain', 'Moderate Rain', 'Heavy Rain / Storm'],
    right=False,
)
print('\nweather category counts (March 2026):')
print(wd['weather_cat'].value_counts().reindex(['Clear', 'Light Rain', 'Moderate Rain', 'Heavy Rain / Storm']).to_string())
print('\nheavy count', int((wd['rain_mm'] >= 25).sum()))
print('\nmoderate or heavier days:')
print(wd.loc[wd['rain_mm'] >= 10, ['date', 'rain_mm', 'wind_gust_max']].to_string(index=False))
print('\ndaily rain summary')
print(wd['rain_mm'].describe().to_string())
print('\nhourly adverse candidates (rain>=10):', int((march['rain_mm'] >= 10).sum()))
print(march.loc[march['rain_mm'] >= 10, ['timestamp_utc', 'date', 'hour', 'rain_mm', 'wind_gust']].to_string(index=False))
