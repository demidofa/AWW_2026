# =============================================================================
# Верификация и анализ архива 45 скважин — Березовское месторождение
# =============================================================================
# Архив:  archive/Well_*.csv  + archive/wells_register.csv
# Цель:   собрать данные, найти ошибки, восстановить целевую Well_34
#         по соседям через Random Forest
# =============================================================================


# %%  [0] Импорты и настройки
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Папка с архивом — рядом со скриптом
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')
print(f"BASE_DIR    = {BASE_DIR}")
print(f"ARCHIVE_DIR = {ARCHIVE_DIR}")

# Целевая скважина для восстановления
TARGET_WELL = 'Well_34'

# Стандартная цветовая схема
COLOR_OK    = '#2E75B6'
COLOR_GREEN = '#1B7A4E'
COLOR_RED   = '#d32f2f'
COLOR_ORANGE = '#F57C00'
COLOR_PURPLE = '#6A0DAD'
ZONE_COLORS = {'Зона_А': '#2E75B6', 'Зона_Б': '#1B7A4E', 'Зона_В': '#d32f2f'}
print("OK")


# %%  [1a] Консолидация архива — словарь алиасов
# В архиве разные стили: советский (cp1251, кириллица, ;, запятая),
# переходный (utf-8, MixedCase, ;), современные (utf-8, ,). Приводим
# названия колонок к единому виду через словарь синонимов.
COLUMN_ALIASES = {
    'well_id':          ['well_id', 'well', 'скважина'],
    'x_m':              ['x_m', 'x'],
    'y_m':              ['y_m', 'y'],
    'zone':             ['zone', 'зона'],
    'data_type':        ['data_type', 'type', 'тип'],
    'depth_m':          ['depth_m', 'глубина', 'depth'],
    'porosity':         ['porosity', 'пористость', 'phi'],
    'permeability_mD':  ['permeability_mD', 'permeability', 'perm_md',
                          'perm', 'проницаемость', 'k_md'],
    'density_gcc':      ['density_gcc', 'density', 'плотность', 'rho'],
    'water_saturation': ['water_saturation', 'sw', 'кв'],
    'oil_saturation':   ['oil_saturation', 'so', 'кн'],
    'lab_id':           ['lab_id', 'lab', 'лаборатория'],
    'year':             ['year', 'год'],
    'PZ':               ['pz', 'rt'],
    'PS':               ['ps', 'sp'],
    'GK':               ['gk', 'gr', 'gamma'],
    'NGK':              ['ngk', 'nphi', 'neutron'],
}
ALIAS_TO_CANON = {a.lower().strip(): c
                   for c, aliases in COLUMN_ALIASES.items()
                   for a in aliases}


def normalize_columns(df):
    """Приводит названия колонок к каноническому виду."""
    new = {}
    for c in df.columns:
        key = c.lower().strip().lstrip('\ufeff')
        new[c] = ALIAS_TO_CANON.get(key, c)
    return df.rename(columns=new)


def read_well_file(path):
    """Читает один файл, перебирая стили чтения. Возвращает (df, opts).

    Выбирает вариант с максимальным числом числовых колонок —
    защита от случая, когда decimal неправильно выбран и числа стали строками.
    """
    candidates = []
    for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
        for sep in [',', ';']:
            for dec in ['.', ',']:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc,
                                      decimal=dec, low_memory=False)
                    if df.shape[1] < 3:
                        continue
                    cols_lower = [c.lower().strip().lstrip('\ufeff')
                                  for c in df.columns]
                    if not any(c in ['well_id', 'well', 'скважина']
                                for c in cols_lower):
                        continue
                    n_num = sum(1 for c in df.columns
                                 if pd.api.types.is_numeric_dtype(df[c]))
                    candidates.append((n_num, df, enc, sep, dec))
                except Exception:
                    continue
    if not candidates:
        raise RuntimeError(f"Не удалось прочитать {path}")
    candidates.sort(key=lambda t: -t[0])
    _, df_best, enc, sep, dec = candidates[0]
    df_best = normalize_columns(df_best)
    return df_best, {'encoding': enc, 'sep': sep, 'decimal': dec}


# %%  [1b] Чтение реестра и всех файлов архива
register = pd.read_csv(os.path.join(ARCHIVE_DIR, 'wells_register.csv'))
print(f"Реестр: {len(register)} скважин")
print(f"  с файлом: {register['file_in_archive'].sum()}")
print(f"  без файла: {(~register['file_in_archive']).sum()}")

all_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, 'Well_*.csv')))
print(f"\nФайлов в архиве: {len(all_files)}\n")

dataframes = []
read_log = []
for path in all_files:
    well_name = os.path.basename(path).replace('.csv', '')
    df, opts = read_well_file(path)
    if 'data_type' in df.columns:
        df['data_type'] = df['data_type'].astype(str).str.lower().str.strip()
    dataframes.append(df)
    read_log.append({'well': well_name, 'rows': len(df), **opts})

log_df = pd.DataFrame(read_log)
print("Стили файлов в архиве:")
print(log_df.groupby(['encoding', 'sep', 'decimal']).size()
      .reset_index(name='count').to_string(index=False))


# %%  [1c] Объединение и разделение на КЕРН / ГИС
master = pd.concat(dataframes, ignore_index=True)

# Конвертация числовых колонок (на случай если decimal оставил строки)
NUM_COLS = ['x_m', 'y_m', 'depth_m', 'porosity', 'permeability_mD',
            'density_gcc', 'water_saturation', 'oil_saturation', 'year',
            'PZ', 'PS', 'GK', 'NGK']
for c in NUM_COLS:
    if c in master.columns:
        master[c] = pd.to_numeric(master[c], errors='coerce')

# Разделение
core_cols = ['well_id', 'x_m', 'y_m', 'zone', 'depth_m', 'porosity',
             'permeability_mD', 'density_gcc', 'water_saturation',
             'oil_saturation', 'lab_id', 'year']
gis_cols = ['well_id', 'x_m', 'y_m', 'zone', 'depth_m',
            'PZ', 'PS', 'GK', 'NGK']

df_core = master[master['data_type'] == 'core'][core_cols].copy()
df_gis  = master[master['data_type'] == 'gis'][gis_cols].copy()
df_gis = df_gis.rename(columns={'depth_m': 'DEPTH'})

# Совместимость со старым скриптом: 'геол_зона' как алиас 'zone'
df_core['геол_зона'] = df_core['zone']

print(f"\nКЕРН: {len(df_core)} образцов по {df_core['well_id'].nunique()} скважинам")
print(f"ГИС:  {len(df_gis)} точек по {df_gis['well_id'].nunique()} скважинам")
print(f"\nСкважин в керне:\n{sorted(df_core['well_id'].unique())}")


# %%  [1d] Описательная статистика керна
df_core.describe().round(4)


# %%  [2] EDA — распределения параметров керна
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f'Распределения параметров керна — {df_core["well_id"].nunique()} скважин',
             fontsize=14, fontweight='bold')

params = ['porosity', 'permeability_mD', 'density_gcc',
          'water_saturation', 'oil_saturation']
colors = ['#2E75B6', '#1B7A4E', '#8B1A1A', '#B87333', '#6A0DAD']
limits = {
    'porosity':         (0, 0.4),
    'permeability_mD':  (0, None),
    'density_gcc':      (2.0, 3.0),
    'water_saturation': (0, 1),
    'oil_saturation':   (0, 1),
}

for i, (col, color) in enumerate(zip(params, colors)):
    ax = axes[i // 3, i % 3]
    data = df_core[col].dropna()
    if col == 'permeability_mD':
        ax.hist(np.log10(data[data > 0]), bins=40, color=color,
                alpha=0.8, edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', lw=2, label='k=0')
        ax.set_xlabel('log₁₀(k), мД')
    else:
        ax.hist(data, bins=40, color=color, alpha=0.8, edgecolor='white')
        lo, hi = limits[col]
        if lo is not None:
            ax.axvline(lo, color='red', linestyle='--', lw=2)
        if hi is not None:
            ax.axvline(hi, color='red', linestyle='--', lw=2)
        ax.set_xlabel(col)
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Частота')

# Scatter φ vs log(k) по лабораториям
ax6 = axes[1, 2]
valid = df_core[(df_core['porosity'].between(0, 0.4)) &
                 (df_core['permeability_mD'] > 0)]
for lab, grp in valid.groupby('lab_id'):
    ax6.scatter(grp['porosity'], np.log10(grp['permeability_mD']),
                alpha=0.4, s=15, label=lab)
ax6.set_xlabel('Пористость')
ax6.set_ylabel('log₁₀(k)')
ax6.legend()
ax6.set_title('Пористость vs Проницаемость', fontweight='bold')

plt.tight_layout()
plt.show()


# %%  [3a] Физичность — пористость
flags = pd.DataFrame(index=df_core.index)
flags['ошибка_пористость'] = ~df_core['porosity'].between(0, 0.4)
n = flags['ошибка_пористость'].sum()
print(f"Пористость вне [0, 0.4]: {n} записей")
if n > 0:
    print(df_core[flags['ошибка_пористость']]
          [['well_id', 'depth_m', 'porosity', 'lab_id']])


# %%  [3b] Физичность — проницаемость
flags['ошибка_проницаемость'] = df_core['permeability_mD'] < 0
n = flags['ошибка_проницаемость'].sum()
print(f"Отрицательная проницаемость: {n} записей")
if n > 0:
    print(df_core[flags['ошибка_проницаемость']]
          [['well_id', 'depth_m', 'permeability_mD']])


# %%  [3c] Физичность — плотность
flags['ошибка_плотность'] = ~df_core['density_gcc'].between(2.0, 3.0)
n = flags['ошибка_плотность'].sum()
print(f"Плотность вне [2.0, 3.0]: {n} записей")
if n > 0:
    print(df_core[flags['ошибка_плотность']]
          [['well_id', 'depth_m', 'density_gcc']])


# %%  [3d] Физичность — баланс насыщенностей
sat_sum = df_core['water_saturation'] + df_core['oil_saturation']
flags['ошибка_насыщенность'] = sat_sum > 1.001
n = flags['ошибка_насыщенность'].sum()
print(f"Sw + So > 1: {n} записей")
if n > 0:
    bad = df_core[flags['ошибка_насыщенность']][
        ['well_id', 'depth_m', 'water_saturation', 'oil_saturation']].copy()
    bad['сумма'] = sat_sum[flags['ошибка_насыщенность']].round(4)
    print(bad)


# %%  [4a] Дубли
key_cols = ['well_id', 'depth_m', 'porosity', 'permeability_mD', 'density_gcc']
flags['дубль'] = df_core.duplicated(subset=key_cols, keep=False)
n = flags['дубль'].sum()
print(f"Дублей: {n} строк ({n//2} пар)")
if n > 0:
    print(df_core[flags['дубль']][key_cols].sort_values(key_cols).head(10))


# %%  [4b] Смещение столбца (Column shift)
flags['смещение_столбца'] = df_core['density_gcc'] > 10
n = flags['смещение_столбца'].sum()
print(f"Смещение столбца (ρ > 10): {n} записей")
if n > 0:
    print(df_core[flags['смещение_столбца']]
          [['well_id', 'depth_m', 'density_gcc', 'permeability_mD']])


# %%  [5a] Лаборатории — описательная статистика
lab_summary = (df_core.groupby('lab_id')['porosity']
                .agg(['count', 'mean', 'median', 'std']).round(4))
print(lab_summary)


# %%  [5b] Лаборатории — визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Сравнение лабораторий по пористости',
             fontsize=13, fontweight='bold')

labs = sorted(df_core['lab_id'].dropna().unique())
lab_colors = ['#2E75B6', '#1B7A4E', '#d32f2f']
data_by_lab = [df_core[df_core['lab_id'] == lab]['porosity'].dropna().values
               for lab in labs]

bp = axes[0].boxplot(data_by_lab, tick_labels=labs, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], lab_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].axhline(df_core['porosity'].median(), color='gray',
                linestyle='--', alpha=0.7, label='Медиана общая')
axes[0].set_title('Boxplot')
axes[0].legend()

for lab, color in zip(labs, lab_colors):
    s = df_core[df_core['lab_id'] == lab]['porosity'].dropna()
    axes[1].hist(s, bins=30, alpha=0.55,
                 label=f'{lab} (med={s.median():.3f})',
                 color=color, density=True)
axes[1].set_title('Распределения')
axes[1].legend()
plt.tight_layout()
plt.show()


# %%  [5c] Лаборатории — статистические тесты
groups = [df_core[df_core['lab_id'] == lab]['porosity'].dropna().values
          for lab in labs]
stat, p = stats.kruskal(*groups)
print(f"Крускал-Уоллис: H={stat:.4f}, p={p:.6f} — "
      f"{'ЗНАЧИМО ***' if p < 0.001 else 'значимо *' if p < 0.05 else 'не значимо'}\n")

for i in range(len(labs)):
    for j in range(i + 1, len(labs)):
        _, p_mw = stats.mannwhitneyu(groups[i], groups[j],
                                      alternative='two-sided')
        sig = ('***' if p_mw < 0.001 else
               '**' if p_mw < 0.01 else
               '*' if p_mw < 0.05 else 'н.з.')
        print(f"  {labs[i]} vs {labs[j]}: p={p_mw:.6f} {sig}")
print()

overall_med = df_core['porosity'].median()
for lab in labs:
    med = df_core[df_core['lab_id'] == lab]['porosity'].median()
    print(f"  {lab}: медиана={med:.4f}  "
          f"({(med - overall_med) / overall_med * 100:+.1f}% от общей)")


# %%  [6a] Итоговый отчёт по верификации керна — сводка
all_flags = {
    'ошибка_пористость':    flags['ошибка_пористость'],
    'ошибка_проницаемость': flags['ошибка_проницаемость'],
    'ошибка_плотность':     flags['ошибка_плотность'],
    'ошибка_насыщенность':  flags['ошибка_насыщенность'],
    'смещение_столбца':     flags['смещение_столбца'],
    'дубль':                flags['дубль'],
}

report = df_core.copy()
for name, flag in all_flags.items():
    report[name] = flag

report['список_ошибок'] = report.apply(
    lambda row: '; '.join([k for k, v in all_flags.items() if v.loc[row.name]]),
    axis=1)
report['есть_ошибка'] = report['список_ошибок'] != ''

total = len(report)
n_errors = int(report['есть_ошибка'].sum())
n_clean = total - n_errors

print(f"Всего:       {total}")
print(f"С ошибками:  {n_errors} ({n_errors/total*100:.1f}%)")
print(f"Чистых:      {n_clean} ({n_clean/total*100:.1f}%)\n")
for name, flag in all_flags.items():
    if flag.sum() > 0:
        print(f"  {name}: {int(flag.sum())}")


# %%  [6b] Сводка — визуализация
counts = {k: int(v.sum()) for k, v in all_flags.items() if v.sum() > 0}

if not counts:
    # Если ошибок нет — показываем только pie с одним сектором
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Результаты верификации керна (45 скважин)',
                 fontsize=13, fontweight='bold')
    ax.pie([n_clean], labels=[f'Чистые ({n_clean})'],
           colors=['#2E75B6'], autopct='%1.1f%%',
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title('Качество данных — ошибок не найдено')
    plt.tight_layout()
    plt.show()
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Результаты верификации керна (45 скважин)',
                 fontsize=13, fontweight='bold')

    axes[0].pie([n_clean, n_errors],
                labels=[f'Чистые ({n_clean})', f'Аномалии ({n_errors})'],
                colors=['#2E75B6', '#d32f2f'], autopct='%1.1f%%',
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title('Качество данных')

    axes[1].barh(list(counts.keys()), list(counts.values()),
                 color=['#d32f2f', '#e64a19', '#7B1FA2',
                        '#1565C0', '#1B5E20', '#F57C00'][:len(counts)])
    axes[1].set_xlabel('Количество записей')
    axes[1].set_title('Типы ошибок')

    plt.tight_layout()
    plt.show()


# %%  [7a] Пространственный анализ — координаты из реестра
# Координаты берём из реестра — там все 45 скважин включая пропущенные
well_coords = {row['well_id']: (row['x_m'], row['y_m'])
               for _, row in register.iterrows()}

clean = report[~report['есть_ошибка']].copy()
well_stats = clean.groupby('well_id').agg(
    por_med=('porosity', 'median'),
    por_std=('porosity', 'std'),
    perm_med=('permeability_mD', 'median'),
    n=('porosity', 'count'),
).reset_index()
well_stats['x'] = well_stats['well_id'].map({k: v[0] for k, v in well_coords.items()})
well_stats['y'] = well_stats['well_id'].map({k: v[1] for k, v in well_coords.items()})
well_stats['zone'] = well_stats['well_id'].map(
    df_core[['well_id', 'геол_зона']].drop_duplicates()
        .set_index('well_id')['геол_зона'])

print(f"Чистых записей: {len(clean)}")
print(f"Скважин в анализе: {len(well_stats)}")
print(well_stats[['well_id', 'x', 'y', 'zone', 'por_med', 'n']]
      .to_string(index=False))


# %%  [7b] Карты скважин — три карты рядом
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Пространственное расположение 45 скважин',
             fontsize=14, fontweight='bold')

# Карта 1: все скважины с зонами + целевая Well_34
ax = axes[0]
for zone, grp in well_stats.groupby('zone'):
    ax.scatter(grp['x'], grp['y'], c=ZONE_COLORS[zone],
               s=grp['n'] * 6, label=zone,
               edgecolors='white', linewidth=1.5, zorder=3)
# Скважины без данных
no_data_wells = register[~register['file_in_archive']]
ax.scatter(no_data_wells['x_m'], no_data_wells['y_m'],
           c='lightgray', s=80, marker='s', edgecolors='gray',
           linewidth=1, label='нет данных', zorder=2)
# Целевая Well_34
tgt = register[register['well_id'] == TARGET_WELL].iloc[0]
ax.scatter(tgt['x_m'], tgt['y_m'], s=350, c='gold', marker='*',
           edgecolor='black', linewidth=2,
           label=f'{TARGET_WELL} (целевая)', zorder=10)

# Подписи
for _, row in register.iterrows():
    label = row['well_id'].replace('Well_', '')
    color = '#444' if row['file_in_archive'] else '#999'
    ax.annotate(label, (row['x_m'], row['y_m']),
                textcoords='offset points', xytext=(7, 5),
                fontsize=7.5, color=color)

ax.set_xlabel('X, м')
ax.set_ylabel('Y, м')
ax.set_title(f'Карта скважин (всего {len(register)}, '
             f'с данными {len(well_stats)})')
ax.legend(fontsize=8, loc='upper left')
ax.set_aspect('equal')

# Карта 2: медианная пористость по площади
ax2 = axes[1]
sc = ax2.scatter(well_stats['x'], well_stats['y'], c=well_stats['por_med'],
                  cmap='RdYlGn', s=200, edgecolors='white', linewidth=1.5,
                  vmin=0.06, vmax=0.16)
plt.colorbar(sc, ax=ax2, label='Медиана пористости', shrink=0.8)
# Целевая
ax2.scatter(tgt['x_m'], tgt['y_m'], s=350, c='gold', marker='*',
            edgecolor='black', linewidth=2, zorder=10)
for _, row in well_stats.iterrows():
    ax2.annotate(f"{row['well_id'].replace('Well_','')}\n{row['por_med']:.3f}",
                 (row['x'], row['y']),
                 textcoords='offset points', xytext=(7, 4), fontsize=6.5)
ax2.set_xlabel('X, м')
ax2.set_ylabel('Y, м')
ax2.set_title('Медианная пористость по площади')
ax2.set_aspect('equal')

# Карта 3: распределение ошибок
ax3 = axes[2]
anomaly_counts = report[report['есть_ошибка']].groupby('well_id').size() \
                   .reset_index(name='n_errors')
anomaly_counts['x'] = anomaly_counts['well_id'].map(
    {k: v[0] for k, v in well_coords.items()})
anomaly_counts['y'] = anomaly_counts['well_id'].map(
    {k: v[1] for k, v in well_coords.items()})

# Все скважины серым фоном
for wid, (wx, wy) in well_coords.items():
    if wid in well_stats['well_id'].values:
        ax3.scatter(wx, wy, c='lightgray', s=80,
                    edgecolors='gray', linewidth=1, zorder=2)

if len(anomaly_counts) > 0:
    sc3 = ax3.scatter(anomaly_counts['x'], anomaly_counts['y'],
                       c=anomaly_counts['n_errors'], cmap='Reds',
                       s=anomaly_counts['n_errors'] * 80,
                       edgecolors='darkred', linewidth=1.5,
                       vmin=0, zorder=3)
    plt.colorbar(sc3, ax=ax3, label='Кол-во ошибок', shrink=0.8)
    for _, row in anomaly_counts.iterrows():
        ax3.annotate(f"{row['well_id'].replace('Well_','')}\n({int(row['n_errors'])})",
                     (row['x'], row['y']),
                     textcoords='offset points', xytext=(7, 5),
                     fontsize=7, color='darkred', fontweight='bold')

ax3.set_xlabel('X, м')
ax3.set_ylabel('Y, м')
ax3.set_title('Распределение ошибок по скважинам')
ax3.set_aspect('equal')

plt.tight_layout()
plt.show()


# %%  [7c] Параметры по геологическим зонам
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Параметры по геологическим зонам (чистые данные)',
             fontsize=13, fontweight='bold')

zones = sorted(clean['геол_зона'].unique())
params_zone = [('porosity', 'Пористость'),
               ('permeability_mD', 'Проницаемость, мД'),
               ('density_gcc', 'Плотность, г/см³')]

for idx, (col, label) in enumerate(params_zone):
    ax = axes[idx]
    data_by_zone = [clean[clean['геол_зона'] == z][col].dropna().values
                    for z in zones]
    bp = ax.boxplot(data_by_zone, tick_labels=zones, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, zone in zip(bp['boxes'], zones):
        patch.set_facecolor(ZONE_COLORS[zone])
        patch.set_alpha(0.6)
    if col == 'permeability_mD':
        ax.set_yscale('log')
    stat, p = stats.kruskal(*data_by_zone)
    sig = ('***' if p < 0.001 else '**' if p < 0.01 else
           '*' if p < 0.05 else 'н.з.')
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel(f'Крускал-Уоллис p={p:.4f} {sig}')

plt.tight_layout()
plt.show()

print("Медианы по зонам:")
print(clean.groupby('геол_зона')
      [['porosity', 'permeability_mD', 'density_gcc']].median().round(4))


# %%  [7d] Вариограмма пористости по парам скважин
pairs = []
for i in range(len(well_stats)):
    for j in range(i + 1, len(well_stats)):
        r1 = well_stats.iloc[i]
        r2 = well_stats.iloc[j]
        dist = ((r1['x'] - r2['x'])**2 + (r1['y'] - r2['y'])**2)**0.5
        gamma = 0.5 * (r1['por_med'] - r2['por_med'])**2
        diff = abs(r1['por_med'] - r2['por_med'])
        pairs.append({'dist': dist, 'gamma': gamma, 'diff': diff,
                      'pair': f"{r1['well_id']} — {r2['well_id']}",
                      'por1': r1['por_med'], 'por2': r2['por_med']})
pairs_df = pd.DataFrame(pairs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Пространственная связность данных',
             fontsize=13, fontweight='bold')

ax1 = axes[0]
ax1.scatter(pairs_df['dist'], pairs_df['gamma'], alpha=0.5,
            color='#2E75B6', s=30, zorder=3)
# Сглаженная кривая по бинам расстояний
bins = np.linspace(0, pairs_df['dist'].max(), 12)
bin_centers = (bins[:-1] + bins[1:]) / 2
binned_gamma = []
for k in range(len(bins) - 1):
    m = (pairs_df['dist'] >= bins[k]) & (pairs_df['dist'] < bins[k + 1])
    if m.sum() > 3:
        binned_gamma.append(pairs_df.loc[m, 'gamma'].mean())
    else:
        binned_gamma.append(np.nan)
ax1.plot(bin_centers, binned_gamma, 'o-', color='#d32f2f',
         lw=2, ms=8, label='среднее по бинам')
ax1.set_xlabel('Расстояние, м')
ax1.set_ylabel('γ(h) = 0.5·(φ₁−φ₂)²')
ax1.set_title('Вариограмма пористости')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
sc = ax2.scatter(pairs_df['dist'], pairs_df['diff'],
                  c=pairs_df['diff'], cmap='RdYlGn_r', s=30, vmin=0, vmax=0.06)
plt.colorbar(sc, ax=ax2, label='|φ₁ − φ₂|')
z = np.polyfit(pairs_df['dist'], pairs_df['diff'], 1)
xline = np.linspace(pairs_df['dist'].min(), pairs_df['dist'].max(), 100)
ax2.plot(xline, np.polyval(z, xline), 'r--', lw=1.5, alpha=0.7, label='Тренд')
ax2.set_xlabel('Расстояние, м')
ax2.set_ylabel('|φ₁ − φ₂|')
ax2.set_title('Сходство пористости vs расстояние')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Топ-5 наиболее похожих пар:")
top = pairs_df.nsmallest(5, 'diff')[['pair', 'dist', 'por1', 'por2', 'diff']].round(4)
top.columns = ['Пара', 'Расстояние, м', 'φ скв.1', 'φ скв.2', 'Разница']
print(top.to_string(index=False))
print("\nТоп-5 наиболее различающихся пар:")
bot = pairs_df.nlargest(5, 'diff')[['pair', 'dist', 'por1', 'por2', 'diff']].round(4)
bot.columns = ['Пара', 'Расстояние, м', 'φ скв.1', 'φ скв.2', 'Разница']
print(bot.to_string(index=False))


# %%  [7e] Профили пористости по глубине у ближайших к Well_34 соседей
# Находим ближайших соседей Well_34 и сравниваем их профили
x34, y34 = well_coords[TARGET_WELL]
neighbor_dists = [(w, np.sqrt((well_coords[w][0] - x34)**2 +
                               (well_coords[w][1] - y34)**2))
                  for w in well_stats['well_id']]
neighbor_dists.sort(key=lambda t: t[1])
print(f"\nWell_34: x={x34}, y={y34}")
print("Ближайшие 4 соседа:")
for w, d in neighbor_dists[:4]:
    print(f"  {w}: {d:.0f} м")

pairs_to_plot = [(neighbor_dists[0][0], neighbor_dists[1][0]),
                 (neighbor_dists[2][0], neighbor_dists[3][0])]

fig, axes = plt.subplots(1, 2, figsize=(12, 7))
fig.suptitle(f'Профили пористости по глубине — ближайшие к {TARGET_WELL}',
             fontsize=12, fontweight='bold')
pair_colors = ['#2E75B6', '#d32f2f']

for ax, (w1, w2) in zip(axes, pairs_to_plot):
    dist_m = int(np.sqrt((well_coords[w1][0] - well_coords[w2][0])**2 +
                          (well_coords[w1][1] - well_coords[w2][1])**2))
    for well, color in zip([w1, w2], pair_colors):
        d = clean[clean['well_id'] == well].sort_values('depth_m')
        ax.scatter(d['porosity'], d['depth_m'], c=color,
                   alpha=0.45, s=20, label=well)
        if len(d) > 5:
            smoothed = uniform_filter1d(d['porosity'].values, size=5)
            ax.plot(smoothed, d['depth_m'].values, color=color,
                    lw=2, alpha=0.85)
    ax.set_xlabel('Пористость, д.е.')
    ax.set_ylabel('Глубина, м')
    ax.set_title(f'{w1} vs {w2} (расстояние {dist_m} м)')
    ax.invert_yaxis()
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# БЛОК [8] — АНАЛИЗ ГИС
# =============================================================================

# %%  [8a] ГИС — обзор покрытия по скважинам
print(f"ГИС: {len(df_gis)} точек по {df_gis['well_id'].nunique()} скважинам")

cov = df_gis.groupby('well_id').agg(
    n_points=('DEPTH', 'count'),
    pz_pct=('PZ', lambda s: s.notna().mean() * 100),
    ps_pct=('PS', lambda s: s.notna().mean() * 100),
    gk_pct=('GK', lambda s: s.notna().mean() * 100),
    ngk_pct=('NGK', lambda s: s.notna().mean() * 100),
).round(1)
print("\nПокрытие ГИС-кривых по скважинам (первые 10):")
print(cov.head(10))
print(f"\nСреднее покрытие по всем: "
      f"{cov[['pz_pct','ps_pct','gk_pct','ngk_pct']].mean().to_dict()}")


# %%  [8b] ГИС — корреляции между кривыми
# Берём строки где все 4 кривые доступны
gis_full = df_gis.dropna(subset=['PZ', 'PS', 'GK', 'NGK']).copy()
print(f"Точек с полным набором: {len(gis_full)}")

corr = gis_full[['PZ', 'PS', 'GK', 'NGK']].corr()
print("\nМатрица корреляций (Пирсон):")
print(corr.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Тепловая карта
im = axes[0].imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))
axes[0].set_xticklabels(corr.columns)
axes[0].set_yticklabels(corr.columns)
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f'{corr.values[i,j]:.2f}',
                     ha='center', va='center',
                     color='white' if abs(corr.values[i,j]) > 0.5 else 'black')
plt.colorbar(im, ax=axes[0], fraction=0.046)
axes[0].set_title('Корреляции ГИС-кривых')

# Scatter самой сильной пары (вне диагонали)
m_abs = corr.abs().where(~np.eye(4, dtype=bool))
imax = m_abs.stack().idxmax()
a, b = imax
# Берём подвыборку для читаемости
sample = gis_full.sample(min(3000, len(gis_full)), random_state=42)
axes[1].scatter(sample[a], sample[b], s=8, alpha=0.3, c='#2E75B6')
axes[1].set_xlabel(a)
axes[1].set_ylabel(b)
axes[1].set_title(f'{a} vs {b} (r = {corr.loc[a, b]:.2f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%  [8c] ГИС — планшет одной скважины (Well_45 как пример)
example_well = 'Well_45' if 'Well_45' in df_gis['well_id'].values \
               else df_gis['well_id'].iloc[0]
gw = df_gis[df_gis['well_id'] == example_well].sort_values('DEPTH')
print(f"Планшет {example_well}: {len(gw)} точек, "
      f"глубина {gw['DEPTH'].min():.0f}-{gw['DEPTH'].max():.0f} м")

fig, axes = plt.subplots(1, 4, figsize=(14, 9), sharey=True)
curves = [('PZ', 'Ом·м', 'log'),
          ('PS', 'мВ', 'linear'),
          ('GK', 'мкР/ч', 'linear'),
          ('NGK', 'усл.ед.', 'linear')]
for ax, (curve, units, scale) in zip(axes, curves):
    d = gw[['DEPTH', curve]].dropna()
    ax.plot(d[curve], d['DEPTH'], lw=0.6, color='#333')
    ax.set_xlabel(f'{curve}, {units}')
    if scale == 'log' and (d[curve] > 0).all():
        ax.set_xscale('log')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Глубина, м')
fig.suptitle(f'Планшет каротажа {example_well}', fontweight='bold')
plt.tight_layout()
plt.show()


# %%  [8d] ГИС — проверка физической согласованности кривых
# Считаем кол-во аномалий по простым правилам
gis_anomalies = pd.DataFrame(index=df_gis.index)
gis_anomalies['neg_PZ']   = df_gis['PZ'] < 0
gis_anomalies['spike_GK'] = df_gis['GK'] > 16
gis_anomalies['out_NGK']  = (df_gis['NGK'] < 80) | (df_gis['NGK'] > 250)

print("Аномалии ГИС по всем скважинам:")
for col in ['neg_PZ', 'spike_GK', 'out_NGK']:
    print(f"  {col:12s}: {int(gis_anomalies[col].sum())}")


# =============================================================================
# БЛОК [9] — ПЕТРОФИЗИКА ПО ФОРМУЛАМ (φ_N и Sw по Архи)
# =============================================================================

# %%  [9a] Калибровка NGK_clean / NGK_clay по керну
# Сопоставляем образцы керна с ближайшими точками ГИС по глубине
# (только в скважинах где есть и керн, и ГИС)
matched = []
for wid in df_gis['well_id'].unique():
    if wid not in clean['well_id'].values:
        continue
    g_w = df_gis[df_gis['well_id'] == wid].dropna(subset=['NGK', 'PZ'])
    c_w = clean[clean['well_id'] == wid]
    for _, c_row in c_w.iterrows():
        if len(g_w) == 0:
            continue
        idx = (g_w['DEPTH'] - c_row['depth_m']).abs().idxmin()
        gap = abs(g_w.loc[idx, 'DEPTH'] - c_row['depth_m'])
        if gap < 2.0:
            matched.append({
                'well_id': wid,
                'phi_core': c_row['porosity'],
                'Sw_core':  c_row['water_saturation'],
                'NGK':      g_w.loc[idx, 'NGK'],
                'GK':       g_w.loc[idx, 'GK'] if pd.notna(g_w.loc[idx, 'GK']) else np.nan,
                'PZ':       g_w.loc[idx, 'PZ'],
            })
mp = pd.DataFrame(matched).dropna(subset=['NGK', 'phi_core'])
print(f"Парных образцов керн↔ГИС: {len(mp)}")

# Линейная регрессия φ_core vs NGK → опорные точки NGK_clean и NGK_clay
lr = LinearRegression().fit(mp[['NGK']].values, mp['phi_core'].values)
a_lr, b_lr = lr.intercept_, lr.coef_[0]
print(f"Линейная модель: φ = {a_lr:.3f} + {b_lr:.5f} * NGK")

if b_lr < 0:
    NGK_clean = max(float((0.30 - a_lr) / b_lr), 80)
    NGK_clay  = min(float(-a_lr / b_lr), 250)
else:
    NGK_clean = max(df_gis['NGK'].quantile(0.01), 80)
    NGK_clay  = min(df_gis['NGK'].quantile(0.99), 250)

print(f"NGK_коллектор = {NGK_clean:.1f}, NGK_глина = {NGK_clay:.1f}")


# %%  [9b] Подбор R_w для Архи
# Sw = sqrt(Rw / (φ² · PZ)) — подбираем Rw минимизируя RMSE с керновым Sw
mp_arc = mp.dropna(subset=['PZ', 'Sw_core'])
best_Rw, best_rmse = None, 1e9
for Rw in np.linspace(0.01, 5.0, 500):
    Sw_calc = ((Rw) / (mp_arc['phi_core']**2 * mp_arc['PZ']))**0.5
    Sw_calc = np.clip(Sw_calc, 0, 1)
    rmse = np.sqrt(((Sw_calc - mp_arc['Sw_core'])**2).mean())
    if rmse < best_rmse:
        best_rmse, best_Rw = rmse, Rw
print(f"R_w = {best_Rw:.3f} Ом·м  (RMSE = {best_rmse:.3f})")


# %%  [9c] Расчёт синтетических кривых на всём ГИС
PHI_MAX = 0.30
pf = df_gis.copy()
pf['phi_N'] = PHI_MAX * (NGK_clay - pf['NGK']) / (NGK_clay - NGK_clean)
pf['phi_N'] = pf['phi_N'].clip(0, 0.4)
pf['Sw_archie'] = np.sqrt(best_Rw / (pf['phi_N'].clip(lower=0.01)**2 *
                                       pf['PZ'].clip(lower=0.01)))
pf['Sw_archie'] = pf['Sw_archie'].clip(0, 1)

print("Покрытие синтетических кривых:")
for c in ['phi_N', 'Sw_archie']:
    pct = pf[c].notna().mean() * 100
    print(f"  {c:12s}: {pct:5.1f}% ({pf[c].notna().sum()} из {len(pf)})")

# Валидация
mp['phi_N'] = PHI_MAX * (NGK_clay - mp['NGK']) / (NGK_clay - NGK_clean)
mp['phi_N'] = mp['phi_N'].clip(0, 0.4)
mp_arc['phi_N'] = PHI_MAX * (NGK_clay - mp_arc['NGK']) / (NGK_clay - NGK_clean)
mp_arc['phi_N'] = mp_arc['phi_N'].clip(lower=0.01)
mp_arc['Sw_archie'] = np.sqrt(best_Rw / (mp_arc['phi_N']**2 *
                                          mp_arc['PZ'].clip(lower=0.01)))
mp_arc['Sw_archie'] = mp_arc['Sw_archie'].clip(0, 1)

r_phi = mp[['phi_N', 'phi_core']].corr().iloc[0, 1]
rmse_phi = np.sqrt(((mp['phi_N'] - mp['phi_core'])**2).mean())
r_sw = mp_arc[['Sw_archie', 'Sw_core']].corr().iloc[0, 1]
rmse_sw = np.sqrt(((mp_arc['Sw_archie'] - mp_arc['Sw_core'])**2).mean())

print(f"\nВалидация по {len(mp)} керновым образцам:")
print(f"  φ_N vs φ_керн:      r = {r_phi:+.3f}, RMSE = {rmse_phi:.4f}")
print(f"  Sw_Архи vs Sw_керн: r = {r_sw:+.3f}, RMSE = {rmse_sw:.4f}")


# %%  [9d] Петрофизика — графики
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(mp['phi_core'], mp['phi_N'], s=30, color='#1B7A4E',
                alpha=0.6, edgecolor='white', label=f'r={r_phi:+.2f}')
lo, hi = mp['phi_core'].min(), mp['phi_core'].max()
axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
axes[0].set_xlabel('φ керн')
axes[0].set_ylabel('φ_N по NGK')
axes[0].set_title(f'Пористость: керн vs формула (RMSE={rmse_phi:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(mp_arc['Sw_core'], mp_arc['Sw_archie'], s=30, color='#d32f2f',
                alpha=0.6, edgecolor='white', label=f'r={r_sw:+.2f}')
axes[1].plot([0, 1], [0, 1], 'r--', lw=1.5)
axes[1].set_xlabel('Sw керн')
axes[1].set_ylabel('Sw по Архи')
axes[1].set_title(f'Водонасыщенность: керн vs Архи (RMSE={rmse_sw:.3f})')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# БЛОК [10] — ВОССТАНОВЛЕНИЕ Well_34 ЧЕРЕЗ RANDOM FOREST
# =============================================================================

# %%  [10a] Подготовка обучающей выборки
# Учим только на физически валидных данных, без Lab_C (там bias)
core_train = clean[clean['lab_id'].isin(['Lab_A', 'Lab_B'])].copy()
core_train['log_K'] = np.log10(core_train['permeability_mD'].clip(lower=1e-3))
print(f"Обучающая выборка: {len(core_train)} строк, "
      f"{core_train['well_id'].nunique()} скважин (Lab_A + Lab_B)")

# Координаты целевой
tgt_x, tgt_y = well_coords[TARGET_WELL]
print(f"\n{TARGET_WELL}: x={tgt_x}, y={tgt_y}, "
      f"зона={register[register['well_id']==TARGET_WELL]['zone'].iloc[0]}")


# %%  [10b] LOO-CV: предсказание медианных свойств скважины по координатам
# Для каждой скважины — одно число (медиана) предсказывается по (x, y)
well_stats_train = core_train.groupby('well_id').agg(
    x=('x_m', 'first'),
    y=('y_m', 'first'),
    phi_med=('porosity', 'median'),
    K_med=('permeability_mD', 'median'),
    rho_med=('density_gcc', 'median'),
).reset_index()
print(f"Скважин для CV: {len(well_stats_train)}")

# Leave-One-Out
preds_phi, true_phi_list = [], []
preds_K_log, true_K_log = [], []
preds_rho, true_rho_list = [], []
for i, row in well_stats_train.iterrows():
    train = well_stats_train.drop(i)
    X_tr = train[['x', 'y']].values
    X_te = [[row['x'], row['y']]]

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, train['phi_med'].values)
    preds_phi.append(rf.predict(X_te)[0])
    true_phi_list.append(row['phi_med'])

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, np.log10(train['K_med'].clip(lower=1e-3)).values)
    preds_K_log.append(rf.predict(X_te)[0])
    true_K_log.append(np.log10(max(row['K_med'], 1e-3)))

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, train['rho_med'].values)
    preds_rho.append(rf.predict(X_te)[0])
    true_rho_list.append(row['rho_med'])

r2_phi_well = r2_score(true_phi_list, preds_phi)
r2_K_well   = r2_score(true_K_log, preds_K_log)
r2_rho_well = r2_score(true_rho_list, preds_rho)
print(f"\nLOO-CV качество (медианы по скважинам, предикторы x,y):")
print(f"  Медианная φ:     R² = {r2_phi_well:+.3f}")
print(f"  log10 K медиан.: R² = {r2_K_well:+.3f}")
print(f"  Медианная ρ:     R² = {r2_rho_well:+.3f}")


# %%  [10c] LOO-CV — визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
panels = [(true_phi_list, preds_phi, 'φ медианная', r2_phi_well),
          (true_K_log,    preds_K_log, 'log K медианная', r2_K_well),
          (true_rho_list, preds_rho,  'ρ медианная', r2_rho_well)]
for ax, (tv, pv, name, r2_v) in zip(axes, panels):
    ax.scatter(tv, pv, s=50, alpha=0.7, c='#2E75B6', edgecolor='white')
    lo = min(min(tv), min(pv))
    hi = max(max(tv), max(pv))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax.set_xlabel(f'{name} истина')
    ax.set_ylabel(f'{name} прогноз')
    ax.set_title(f'LOO-CV: {name} (R²={r2_v:.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%  [10d] Финальное предсказание Well_34
# Обучаемся на ВСЕХ скважинах и предсказываем целевую
rf_phi = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
rf_phi.fit(well_stats_train[['x', 'y']].values,
            well_stats_train['phi_med'].values)
phi34_pred = float(rf_phi.predict([[tgt_x, tgt_y]])[0])

rf_K = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
rf_K.fit(well_stats_train[['x', 'y']].values,
          np.log10(well_stats_train['K_med'].clip(lower=1e-3)).values)
K34_pred = float(10 ** rf_K.predict([[tgt_x, tgt_y]])[0])

rf_rho = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
rf_rho.fit(well_stats_train[['x', 'y']].values,
            well_stats_train['rho_med'].values)
rho34_pred = float(rf_rho.predict([[tgt_x, tgt_y]])[0])

print(f"=== ПРОГНОЗ ДЛЯ {TARGET_WELL} (x={tgt_x}, y={tgt_y}) ===")
print(f"  Медианная пористость:  {phi34_pred:.3f}")
print(f"  Медианная K, мД:       {K34_pred:.2f}")
print(f"  Медианная плотность:   {rho34_pred:.3f}")

# Если есть файл с истиной — сравним
truth_path = os.path.join(BASE_DIR, 'Well_34_TRUTH_for_validation.csv')
if os.path.exists(truth_path):
    truth = pd.read_csv(truth_path)
    truth_core = truth[truth['data_type'] == 'core']
    phi34_true = truth_core['porosity'].median()
    K34_true   = truth_core['permeability_mD'].median()
    rho34_true = truth_core['density_gcc'].median()
    print(f"\n=== СРАВНЕНИЕ С ИСТИНОЙ (Well_34_TRUTH_for_validation.csv) ===")
    print(f"{'Параметр':<22} {'Истина':>10} {'Прогноз':>10} {'Ошибка':>10}")
    print(f"  {'Медианная φ':<20} {phi34_true:>10.3f} {phi34_pred:>10.3f} "
          f"{(phi34_pred - phi34_true) / phi34_true * 100:>+9.1f}%")
    print(f"  {'Медианная K, мД':<20} {K34_true:>10.2f} {K34_pred:>10.2f} "
          f"{(K34_pred - K34_true) / K34_true * 100:>+9.1f}%")
    print(f"  {'Медианная ρ, г/см³':<20} {rho34_true:>10.3f} {rho34_pred:>10.3f} "
          f"{(rho34_pred - rho34_true) / rho34_true * 100:>+9.1f}%")
else:
    print(f"\n(файл с истиной не найден: {truth_path})")


# %%  [10e] Карта пористости с интерполяцией + прогноз Well_34
# Строим карту пористости на сетке через RF и накладываем
# точки реальных скважин + Well_34
xx, yy = np.meshgrid(
    np.linspace(register['x_m'].min() - 200, register['x_m'].max() + 200, 60),
    np.linspace(register['y_m'].min() - 200, register['y_m'].max() + 200, 60))
grid = np.column_stack([xx.ravel(), yy.ravel()])
phi_grid = rf_phi.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(11, 8))
cf = ax.contourf(xx, yy, phi_grid, levels=20, cmap='RdYlGn', alpha=0.8)
plt.colorbar(cf, ax=ax, label='Прогноз медианной пористости')

# Скважины с данными
for zone, grp in well_stats.groupby('zone'):
    ax.scatter(grp['x'], grp['y'], c=ZONE_COLORS[zone],
               s=80, label=zone, edgecolors='white', linewidth=1.5, zorder=3)

# Скважины без данных
no_data_wells = register[~register['file_in_archive']]
ax.scatter(no_data_wells['x_m'], no_data_wells['y_m'],
           c='lightgray', s=60, marker='s', edgecolors='gray',
           linewidth=1, label='нет данных', zorder=2)

# Целевая
ax.scatter(tgt_x, tgt_y, s=400, c='gold', marker='*',
           edgecolor='black', linewidth=2,
           label=f'{TARGET_WELL} (прогноз φ={phi34_pred:.3f})', zorder=10)

# Подписи
for _, row in register.iterrows():
    color = '#222' if row['file_in_archive'] else '#888'
    ax.annotate(row['well_id'].replace('Well_', ''),
                (row['x_m'], row['y_m']),
                textcoords='offset points', xytext=(7, 5),
                fontsize=7, color=color)

ax.set_xlabel('X, м')
ax.set_ylabel('Y, м')
ax.set_title(f'Карта прогноза пористости (RF) с положением {TARGET_WELL}',
             fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()


# =============================================================================
# БЛОК [11] — СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

# %%  [11] Сохранение всех выходных файлов
# Сохраняются в папку рядом со скриптом

# 1) Консолидированные данные
df_core.to_csv(os.path.join(BASE_DIR, 'master_core.csv'),
                index=False, encoding='utf-8-sig')
df_gis.to_csv(os.path.join(BASE_DIR, 'master_gis.csv'),
               index=False, encoding='utf-8-sig')

# 2) Отчёт о верификации
anomalies = report[report['есть_ошибка']]
anomalies.to_csv(os.path.join(BASE_DIR, 'anomaly_report.csv'),
                  index=True, encoding='utf-8-sig')
report.to_csv(os.path.join(BASE_DIR, 'full_report.csv'),
               index=True, encoding='utf-8-sig')

# 3) Синтетические петрофизические кривые
pf_out = pf[['well_id', 'x_m', 'y_m', 'DEPTH', 'PZ', 'PS', 'GK', 'NGK',
              'phi_N', 'Sw_archie']]
pf_out.to_csv(os.path.join(BASE_DIR, 'petrophysics_synthetic.csv'),
               index=False, encoding='utf-8-sig')

# 4) Прогноз Well_34
well34_pred = pd.DataFrame([{
    'well_id': TARGET_WELL,
    'x_m': tgt_x, 'y_m': tgt_y,
    'phi_median_predicted': phi34_pred,
    'K_mD_median_predicted': K34_pred,
    'density_median_predicted': rho34_pred,
    'method': 'RandomForest по координатам соседних скважин',
}])
well34_pred.to_csv(os.path.join(BASE_DIR, 'well34_prediction.csv'),
                    index=False, encoding='utf-8-sig')

# 5) Лог чтения архива
log_df.to_csv(os.path.join(BASE_DIR, 'archive_read_log.csv'),
               index=False, encoding='utf-8-sig')

print(f"Сохранено в {BASE_DIR}:")
print(f"  • master_core.csv               ({len(df_core)} строк керна)")
print(f"  • master_gis.csv                ({len(df_gis)} строк ГИС)")
print(f"  • anomaly_report.csv            ({len(anomalies)} аномальных строк)")
print(f"  • full_report.csv               ({len(report)} строк с флагами)")
print(f"  • petrophysics_synthetic.csv    ({len(pf_out)} строк синтетики)")
print(f"  • well34_prediction.csv         (прогноз для {TARGET_WELL})")
print(f"  • archive_read_log.csv          (стили файлов в архиве)")

print(f"\n=== ИТОГ ===")
print(f"  Архив:        {len(register)} скважин в реестре, "
      f"{register['file_in_archive'].sum()} с файлом, "
      f"{(~register['file_in_archive']).sum()} без данных")
print(f"  Керн:         {len(df_core)} образцов, "
      f"{n_errors} с ошибками ({n_errors/total*100:.1f}%)")
print(f"  ГИС:          {len(df_gis)} точек, "
      f"среднее покрытие {cov[['pz_pct','ps_pct','gk_pct','ngk_pct']].mean().mean():.0f}%")
print(f"  Физика:       r(φ, log K) = "
      f"{core_train[['porosity','log_K']].corr().iloc[0,1]:+.2f}, "
      f"r(φ, ρ) = "
      f"{core_train[['porosity','density_gcc']].corr().iloc[0,1]:+.2f}")
print(f"  Петрофизика:  φ_N ↔ керн r = {r_phi:+.2f}, "
      f"Sw_Архи ↔ керн r = {r_sw:+.2f}")
print(f"  {TARGET_WELL}:      прогноз φ = {phi34_pred:.3f}, "
      f"K = {K34_pred:.1f} мД, ρ = {rho34_pred:.3f}")
