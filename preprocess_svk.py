"""Read only data/SvK Excel files; preserve regions, categories, signs and MWh.

Notebook usage:
    from preprocess_svk import load_svk_data
    df_consumption, df_generation, df_losses = load_svk_data()

Run directly to export the three tables to data/clean/SvK.
Missing categories remain NaN. No regional/category aggregation is performed.
Timestamps retain the source's timezone convention (no timezone is inferred).
"""

from pathlib import Path
import re
import warnings

import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN = {
    'Timmätt förbr': 'metered_consumption',
    'Avkopplingsb.': 'interruptible',
    'Avkopplingsbar': 'interruptible',
    'Energilager': 'energy_storage',
    'Timmätta': 'metered',
    'Ospec.': 'unspecified',
    'Vattenkraft': 'hydro',
    'Vindkraft': 'wind',
    'Kärnkraft': 'nuclear',
    'Värmekraft': 'thermal',
    'Solkraft': 'solar',
    'Schablonleverans': 'profiled_supply',
}
SUB = {
    'exkl. avk.last': 'excluding_interruptible_load',
    '>50 MW': 'over_50mw',
    'last': 'load',
    'förbrukning': 'consumption',
    'förluster': 'losses',
    'produktion': 'generation',
    'landbaserad': 'onshore',
    'havsbaserad': 'offshore',
}


def read_svk_file(path):
    """Translate the four header rows and return three separate tables."""
    path = Path(path)
    raw = pd.read_excel(path, sheet_name=0, header=None, engine='xlrd')
    # The first observation is at row 5, not row 6. Parse only below headers.
    body = raw.iloc[4:].copy()
    timestamps = pd.to_datetime(
        body.iloc[:, 0], format='mixed', dayfirst=True, errors='coerce'
    )
    valid = timestamps.notna()
    if not valid.any():
        raise ValueError(f'{path.name}: no timestamp rows found')
    index = pd.DatetimeIndex(timestamps[valid], name='datetime')
    groups = {key: {} for key in ('consumption', 'generation', 'losses')}

    for col in range(1, raw.shape[1]):
        # Ignore entirely empty trailing spreadsheet columns, not real data.
        if raw.iloc[:, col].isna().all():
            continue
        main, sub, zone, unit = (
            '' if pd.isna(v) else str(v).strip()
            for v in raw.iloc[:4, col]
        )
        if not any((main, sub, zone, unit)):
            warnings.warn(
                f'{path.name}: unlabeled Excel column {col + 1} contains values; '
                'excluded because its region/category is unknown (source is unchanged)'
            )
            continue
        if main not in MAIN or sub not in SUB or zone not in {'SE1', 'SE2', 'SE3', 'SE4'}:
            raise ValueError(f'{path.name}: unrecognized header {(main, sub, zone)}')
        if unit != 'MWh':
            raise ValueError(f'{path.name}: unexpected unit {unit!r}')

        if sub == 'förluster':
            group = 'losses'
        elif sub == 'produktion' or (main == 'Vindkraft' and sub in {'landbaserad', 'havsbaserad'}):
            group = 'generation'
        elif main in {'Timmätt förbr', 'Avkopplingsb.', 'Avkopplingsbar'} or sub == 'förbrukning':
            group = 'consumption'
        else:
            raise ValueError(f'{path.name}: cannot classify {(main, sub)}')

        name = '_'.join([zone.lower(), MAIN[main], SUB[sub]])
        if name in groups[group]:
            raise ValueError(f'{path.name}: duplicate column {name}')
        source = body.loc[valid, col]
        numeric = pd.to_numeric(source, errors='coerce')
        invalid = source.notna() & numeric.isna()
        if invalid.any():
            warnings.warn(f'{path.name}: {invalid.sum()} nonnumeric values in {name} became NaN')
        groups[group][name] = numeric.to_numpy()

    return {
        key: pd.DataFrame(columns, index=index).sort_index(kind='stable')
        for key, columns in groups.items()
    }


def load_svk_data(folder=None, start_year=2018, end_year=2026):
    """Return (df_consumption, df_generation, df_losses) across all years.

    Each year must have exactly one timvarden-YYYY*.xls file. Columns are
    aligned by name across years; no missing values or timestamps are dropped.
    Duplicate timestamps are preserved and reported, never silently averaged.
    """
    folder = Path(folder) if folder is not None else ROOT / 'data' / 'SvK'
    files = {}
    for path in sorted(folder.glob('timvarden-*.xls')):
        match = re.match(r'timvarden-(\d{4})', path.name)
        if match and start_year <= int(match[1]) <= end_year:
            year = int(match[1])
            if year in files:
                raise ValueError(f'Multiple source files for {year}: {files[year]}, {path}')
            files[year] = path
    missing = sorted(set(range(start_year, end_year + 1)) - files.keys())
    if missing:
        raise FileNotFoundError(f'Missing SvK files for years {missing} in {folder}')

    collected = {key: [] for key in ('consumption', 'generation', 'losses')}
    for year, path in sorted(files.items()):
        tables = read_svk_file(path)
        for key, frame in tables.items():
            if not (frame.index.year == year).all():
                raise ValueError(f'{path.name}: timestamps outside filename year {year}')
            collected[key].append(frame)
        print(f'{path.name}: {len(tables["consumption"]):,} timestamp rows')

    combined = []
    for key, frames in collected.items():
        frame = pd.concat(frames, axis=0, join='outer', sort=False).sort_index(kind='stable')
        if frame.index.has_duplicates:
            warnings.warn(f'{key}: duplicate timestamps preserved; inspect source time convention')
        combined.append(frame)
    return tuple(combined)


if __name__ == '__main__':
    df_consumption, df_generation, df_losses = load_svk_data()
    output = ROOT / 'data' / 'clean' / 'SvK'
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ('consumption', df_consumption),
        ('generation', df_generation),
        ('losses', df_losses),
    ):
        frame.to_csv(output / f'{name}_2018_2026.csv')
        print(f'{name}: {frame.shape}, {frame.index.min()} through {frame.index.max()}')
