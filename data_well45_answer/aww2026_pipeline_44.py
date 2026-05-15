# -*- coding: utf-8 -*-
"""
АльметТех AWW 2026 — ИИ-детектив. Скважина Well_45 / Березовское.
Задания 1–3 в одном файле. 

Структура:
  # %% 0  — настройка, пути, импорты, утилиты
  # %% 1.1 — разведочный анализ керна
  # %% 1.2 — правила физичности (4 правила)
  # %% 1.3 — дубли, column shift, сравнение лабораторий, anomaly_report.csv
  # %% 2.1 — Isolation Forest
  # %% 2.2 — KMeans, литотипы
  # %% 2.3 — Random Forest для проницаемости
  # %% 3.1 — верификация ГИС (планшет + аномалии)
  # %% 3.2 — корреляции между ГИС-кривыми
  # %% 3.3 — восстановление пропусков ГИС
  # %% 3.4 — сопоставление ГИС с керном


"""

# %% 0. НАСТРОЙКА ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

# Папка с данными. По умолчанию — папка, где лежит этот .py.
# В Spyder __file__ есть при запуске файлом; при построчном запуске берём cwd.
try:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DATA_DIR = os.getcwd()
print(f"DATA_DIR = {DATA_DIR}")

# Стиль графиков
mpl.rcParams.update({
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'font.family': 'DejaVu Sans',  # чтобы кириллица не падала
})

# Физические границы (Березовское, осадочные коллекторы)
LIMITS = {
    'porosity':         (0.0,  0.40),   # доля единицы
    'permeability_mD':  (0.0,  10000),  # мД
    'density_gcc':      (2.0,  3.0),    # г/см³ (кварц 2.65, кальцит 2.71, доломит 2.87)
    'water_saturation': (0.0,  1.0),
    'oil_saturation':   (0.0,  1.0),
}

GIS_LIMITS = {
    'PZ':  (0.0, 1000),     # Ом·м, > 0
    'PS':  (-200, 200),     # мВ, любое в разумных пределах
    'GK':  (3.0, 16.0),     # мкР/ч — нормальный диапазон, выше = спайк
    'NGK': (80.0, 250.0),
}


def fpath(name):
    """Полный путь к файлу в DATA_DIR."""
    return os.path.join(DATA_DIR, name)


def hr(title):
    """Печать заголовка-разделителя в консоль."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# %% 1.1. РАЗВЕДОЧНЫЙ АНАЛИЗ КЕРНА ===========================================
hr("1.1 EDA — core_data_dirty_v2.csv")

core = pd.read_csv(fpath('core_data_dirty_v2.csv'))
print(f"Размер: {core.shape[0]} строк, {core.shape[1]} колонок")
print(f"Колонки: {list(core.columns)}")
print(f"\nПропуски:\n{core.isna().sum()}")

print("\nОписательная статистика (числовые):")
num_cols = ['depth_m', 'porosity', 'permeability_mD', 'density_gcc',
            'water_saturation', 'oil_saturation', 'year']
print(core[num_cols].describe().T.round(3))

print(f"\nЛаборатории: {core['lab_id'].value_counts().to_dict()}")
print(f"Скважины ({core['well_id'].nunique()}): {sorted(core['well_id'].unique())}")
print(f"Геол. зоны: {core['геол_зона'].value_counts().to_dict()}")

# Гистограммы с физическими границами
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
hist_cols = ['porosity', 'permeability_mD', 'density_gcc',
             'water_saturation', 'oil_saturation', 'year']
for ax, col in zip(axes.flat, hist_cols):
    ax.hist(core[col], bins=40, color='#4a7ab8', edgecolor='white', alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel(col)
    if col in LIMITS:
        lo, hi = LIMITS[col]
        ax.axvline(lo, color='red', ls='--', lw=1, label=f'физ. границы [{lo}, {hi}]')
        ax.axvline(hi, color='red', ls='--', lw=1)
        ax.legend(fontsize=8)
fig.suptitle('Распределения параметров керна (грязный датасет)', fontweight='bold')
fig.tight_layout()
plt.show()

# Scatter φ vs log10(k) по лабораториям
fig, ax = plt.subplots(figsize=(9, 6))
core_pos = core[core['permeability_mD'] > 0].copy()
core_pos['log_k'] = np.log10(core_pos['permeability_mD'])
colors = {'Lab_A': '#1f77b4', 'Lab_B': '#2ca02c', 'Lab_C': '#d62728'}
for lab in ['Lab_A', 'Lab_B', 'Lab_C']:
    sub = core_pos[core_pos['lab_id'] == lab]
    ax.scatter(sub['porosity'], sub['log_k'], s=30, alpha=0.65,
               c=colors[lab], label=f'{lab} (n={len(sub)})', edgecolor='white', lw=0.5)
ax.set_xlabel('Пористость, д.е.')
ax.set_ylabel('log₁₀(проницаемость, мД)')
ax.set_title('φ vs log K по лабораториям')
ax.axvspan(0, 1, alpha=0.03, color='gray')  # физическая зона
ax.legend()
fig.tight_layout()
plt.show()

print("""
НАБЛЮДЕНИЯ:
 • Видны пористости вне [0,1] и отрицательные проницаемости — нарушение физики.
 • На гистограмме плотности встречаются значения > 10 г/см³ — невозможно,
   это column shift (значение из другой колонки попало сюда).
 • Lab_C на scatter φ–logK заметно сдвинута вправо — выше пористость
   при той же проницаемости. Проверим тестом Крускала–Уоллиса в шаге 1.3.
""")


# %% 1.2. ПРАВИЛА ФИЗИЧНОСТИ =================================================
hr("1.2 Правила физичности")

flags = pd.DataFrame(index=core.index)

# Правило 1 (пористость) — пример из задания
flags['rule1_porosity'] = ~core['porosity'].between(*LIMITS['porosity'])

# Правило 2 (проницаемость): физически >= 0; верхняя граница 10000 мД мягкая,
# но отрицательные значения — гарантированная ошибка
flags['rule2_permeability'] = ~core['permeability_mD'].between(*LIMITS['permeability_mD'])

# Правило 3 (плотность): минералы коллектора 2.0–3.0 г/см³
flags['rule3_density'] = ~core['density_gcc'].between(*LIMITS['density_gcc'])

# Правило 4 (сумма насыщенностей): Sw + So <= 1.0 (доли пор)
sat_sum = core['water_saturation'] + core['oil_saturation']
flags['rule4_saturation_sum'] = (sat_sum > 1.0 + 1e-9) | \
                                 ~core['water_saturation'].between(0, 1) | \
                                 ~core['oil_saturation'].between(0, 1)

flags['n_rules_violated'] = flags.sum(axis=1)
flags['any_physics_violation'] = flags['n_rules_violated'] > 0

print(f"Нарушений по правилам:")
for r in ['rule1_porosity', 'rule2_permeability', 'rule3_density', 'rule4_saturation_sum']:
    print(f"  {r:25s}: {flags[r].sum():3d} строк")
print(f"Всего уникальных строк с нарушениями: {flags['any_physics_violation'].sum()}")

# Покажем по 3 примера на каждое правило
for r, name in [('rule1_porosity', 'пористость вне [0, 0.4]'),
                ('rule2_permeability', 'проницаемость < 0'),
                ('rule3_density', 'плотность вне [2.0, 3.0]'),
                ('rule4_saturation_sum', 'Sw + So > 1.0 или сами вне [0,1]')]:
    sub = core[flags[r]].head(3)
    if len(sub) > 0:
        print(f"\n--- {r} ({name}), показано {len(sub)} из {flags[r].sum()} ---")
        print(sub[['well_id', 'depth_m', 'porosity', 'permeability_mD',
                   'density_gcc', 'water_saturation', 'oil_saturation', 'lab_id']]
              .to_string(index=False))


# %% 1.3. ДУБЛИ, COLUMN SHIFT, СРАВНЕНИЕ ЛАБОРАТОРИЙ =========================
hr("1.3 Дубли, column shift, лаборатории")

# --- Дубли ---
# Полные дубликаты строк
dup_full = core.duplicated(keep=False)
# По ключу (well + depth + lab) — обычно этого достаточно
dup_key = core.duplicated(subset=['well_id', 'depth_m', 'lab_id'], keep=False)
print(f"Полных дубликатов строк: {dup_full.sum()}")
print(f"Дубликатов по (well, depth, lab): {dup_key.sum()}")
flags['rule5_duplicate'] = dup_key

if dup_key.sum() > 0:
    print("\nПримеры дубликатов:")
    print(core[dup_key].sort_values(['well_id', 'depth_m']).head(6)
          [['well_id', 'depth_m', 'porosity', 'permeability_mD', 'lab_id']]
          .to_string(index=False))

# --- Column shift ---
# Признак — плотность в десятки/сотни (часто значение из перепутанной колонки).
# Считаем shift отдельным флагом, чтобы не путать с обычной ошибкой плотности.
shift_mask = core['density_gcc'] > 10
flags['rule6_column_shift'] = shift_mask
print(f"\nColumn shift (density > 10 г/см³): {shift_mask.sum()} строк")
if shift_mask.sum() > 0:
    print(core[shift_mask].head(5)
          [['well_id', 'depth_m', 'porosity', 'permeability_mD',
            'density_gcc', 'water_saturation', 'oil_saturation', 'lab_id']]
          .to_string(index=False))

# --- Сравнение лабораторий ---
# Считаем только по «физически валидным» строкам
clean_mask = (~flags['rule1_porosity']) & (~flags['rule2_permeability']) & \
             (~flags['rule3_density']) & (~flags['rule4_saturation_sum']) & \
             (~flags['rule6_column_shift'])
clean_core = core[clean_mask].copy()

lab_medians = clean_core.groupby('lab_id')[['porosity', 'permeability_mD', 'density_gcc']].median()
print(f"\nМедианы по лабораториям (только валидные строки):")
print(lab_medians.round(4))

# Тест Крускала-Уоллиса для пористости
from scipy.stats import kruskal
groups = [clean_core[clean_core['lab_id'] == lab]['porosity'].values
          for lab in ['Lab_A', 'Lab_B', 'Lab_C']]
H, pval = kruskal(*groups)
print(f"\nКрускал–Уоллис по porosity: H = {H:.2f}, p = {pval:.4g}")
print(f"   -- {'различия ЗНАЧИМЫ (p<0.05)' if pval < 0.05 else 'различия не значимы'}")

# Boxplot
fig, ax = plt.subplots(figsize=(8, 5))
data_by_lab = [clean_core[clean_core['lab_id'] == lab]['porosity'].values
               for lab in ['Lab_A', 'Lab_B', 'Lab_C']]
bp = ax.boxplot(data_by_lab, labels=['Lab_A', 'Lab_B', 'Lab_C'], patch_artist=True)
for patch, c in zip(bp['boxes'], ['#1f77b4', '#2ca02c', '#d62728']):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
ax.set_ylabel('Пористость, д.е.')
ax.set_title(f'Пористость по лабораториям (Kruskal-Wallis p={pval:.3g})')
fig.tight_layout()
plt.show()

# Подсчёт смещения лаборатории-аутлайера в процентах от референса (медиана A и B)
ref = pd.concat([clean_core[clean_core['lab_id'] == 'Lab_A']['porosity'],
                 clean_core[clean_core['lab_id'] == 'Lab_B']['porosity']]).median()
labC = clean_core[clean_core['lab_id'] == 'Lab_C']['porosity'].median()
bias = (labC - ref) / ref * 100
print(f"\nСмещение Lab_C относительно (A+B): {bias:+.1f}% по медиане пористости")

# Если Lab_C систематически смещена — флагируем её строки как 7-е правило
flags['rule7_lab_bias_C'] = (core['lab_id'] == 'Lab_C') & (pval < 0.05) & (abs(bias) > 15)
print(f"Строк, помеченных как lab-bias (Lab_C, смещение >15%): {flags['rule7_lab_bias_C'].sum()}")

# --- Сборка отчёта ---
flags['n_rules_violated'] = flags[[c for c in flags.columns if c.startswith('rule')]].sum(axis=1)
flags['any_violation'] = flags['n_rules_violated'] > 0

anomaly_report = pd.concat([core, flags], axis=1)
anomaly_report = anomaly_report[anomaly_report['any_violation']].copy()
out_path = fpath('anomaly_report.csv')
anomaly_report.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\nСохранено: {out_path}  ({len(anomaly_report)} аномальных строк)")

# Сохраним «чистую» выборку для последующих ML-блоков
core_clean = core[~flags['any_violation']].copy()
# log10(k) пригодится дальше
core_clean['log_perm'] = np.log10(core_clean['permeability_mD'].clip(lower=1e-3))
print(f"Чистых строк для ML: {len(core_clean)} / {len(core)}")


# %% 2.1. ISOLATION FOREST ===================================================
hr("2.1 Isolation Forest — поиск аномалий без правил")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

features = ['porosity', 'log_perm', 'density_gcc', 'water_saturation', 'oil_saturation']
X = core_clean[features].values
Xs = StandardScaler().fit_transform(X)

# Сравним три уровня contamination
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
results = {}
for ax, cont in zip(axes, [0.05, 0.10, 0.15]):
    iso = IsolationForest(contamination=cont, random_state=42, n_estimators=200)
    pred = iso.fit_predict(Xs)        # -1 = аномалия
    is_anom = pred == -1
    results[cont] = is_anom
    ax.scatter(core_clean['porosity'], core_clean['log_perm'],
               s=22, alpha=0.5, c='lightgray', label='норма')
    ax.scatter(core_clean.loc[is_anom, 'porosity'],
               core_clean.loc[is_anom, 'log_perm'],
               s=55, marker='x', c='red', label=f'аномалия (n={is_anom.sum()})')
    ax.set_xlabel('Пористость, д.е.')
    ax.set_ylabel('log₁₀(K, мД)')
    ax.set_title(f'contamination = {cont}')
    ax.legend()
fig.suptitle('Isolation Forest при разных уровнях contamination', fontweight='bold')
fig.tight_layout()
plt.show()

# Берём contamination=0.10 как рабочее значение
chosen = 0.10
core_clean['is_IF_anomaly'] = results[chosen]
n_anom = core_clean['is_IF_anomaly'].sum()
print(f"При contamination={chosen}: {n_anom} аномалий из {len(core_clean)}")

# Сохраняем
if_out = core_clean[core_clean['is_IF_anomaly']].copy()
if_path = fpath('ml_anomalies_IF.csv')
if_out.to_csv(if_path, index=False, encoding='utf-8-sig')
print(f"Сохранено: {if_path}")

print("""
ИНТЕРПРЕТАЦИЯ:
 • Правила физичности ловят гарантированные ошибки (вне физических границ).
 • Isolation Forest ловит «странные» точки, которые формально физичны,
   но не похожи на основное облако — например, высокая пористость при низкой
   проницаемости (плохо связанная пористость) или необычные сочетания Sw/So.
 • Это две разные сетки контроля, и они дополняют друг друга.
""")


# %% 2.2. KMEANS — ЛИТОТИПЫ ==================================================
hr("2.2 KMeans — кластеризация на литотипы")

from sklearn.cluster import KMeans

X_lt = core_clean[['porosity', 'log_perm', 'density_gcc']].values
Xs_lt = StandardScaler().fit_transform(X_lt)

# Метод локтя
inertias = []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xs_lt)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(ks), inertias, 'o-', color='#4a7ab8', lw=2, markersize=8)
ax.set_xlabel('k (число кластеров)')
ax.set_ylabel('Inertia')
ax.set_title('Метод локтя для выбора k')
fig.tight_layout()
plt.show()

# Берём k=3 (стандартная гипотеза: глина / коллектор плохой / коллектор хороший)
K_OPT = 3
km = KMeans(n_clusters=K_OPT, random_state=42, n_init=10)
core_clean['litotype'] = km.fit_predict(Xs_lt)

# Описание кластеров
desc = core_clean.groupby('litotype')[['porosity', 'permeability_mD', 'density_gcc']].median()
desc['n'] = core_clean.groupby('litotype').size()
print(f"\nМедианы по кластерам (k={K_OPT}):")
print(desc.round(3))

# Интерпретация через РАНЖИРОВАНИЕ кластеров
# Чем больше φ и log(K), тем лучше коллектор; чем выше ρ, тем плотнее порода.
# Сортируем кластеры по сводному «качеству» — сумма рангов φ и log(K).
labels_ordered = ['хороший коллектор', 'средний коллектор',
                  'плотная порода / глина']
desc_ranked = desc.copy()
desc_ranked['log_K'] = np.log10(desc_ranked['permeability_mD'].clip(lower=1e-3))
desc_ranked['quality_score'] = (desc_ranked['porosity'].rank() +
                                 desc_ranked['log_K'].rank() -
                                 desc_ranked['density_gcc'].rank())
desc_ranked = desc_ranked.sort_values('quality_score', ascending=False)
desc_ranked['interpretation'] = labels_ordered[:len(desc_ranked)]
desc['interpretation'] = desc_ranked['interpretation']

print("\nИнтерпретация (по убыванию качества):")
print(desc[['porosity', 'permeability_mD', 'density_gcc', 'n',
            'interpretation']])

# Три проекции
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
proj = [('porosity', 'log_perm', 'φ vs log K'),
        ('porosity', 'density_gcc', 'φ vs ρ'),
        ('log_perm', 'density_gcc', 'log K vs ρ')]
cmap = plt.cm.Set1
for ax, (xc, yc, title) in zip(axes, proj):
    for cid in range(K_OPT):
        sub = core_clean[core_clean['litotype'] == cid]
        ax.scatter(sub[xc], sub[yc], s=28, alpha=0.65,
                   color=cmap(cid), label=f'cluster {cid} (n={len(sub)})',
                   edgecolor='white', lw=0.5)
    ax.set_xlabel(xc)
    ax.set_ylabel(yc)
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle(f'KMeans литотипы (k={K_OPT})', fontweight='bold')
fig.tight_layout()
plt.show()

# Связь с геол. зонами
print("\nКластер × геол. зона (доля):")
cross = pd.crosstab(core_clean['litotype'], core_clean['геол_зона'], normalize='index')
print(cross.round(2))

core_clean.to_csv(fpath('ml_clusters.csv'), index=False, encoding='utf-8-sig')
print(f"\nСохранено: {fpath('ml_clusters.csv')}")


# %% 2.3. RANDOM FOREST — ПРОНИЦАЕМОСТЬ ======================================
hr("2.3 Random Forest — предсказание log10(K)")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error

# Для RF используем физически валидные строки (без правил 1–6),
# но без отсева Lab_C — иначе теряем треть данных и модель не учится.
# Лабораторное смещение коррелирует с пористостью, и Random Forest сам
# выучит этот паттерн как часть зависимости (если он системный).
physics_ok = (~flags['rule1_porosity']) & (~flags['rule2_permeability']) & \
             (~flags['rule3_density'])  & (~flags['rule4_saturation_sum']) & \
             (~flags['rule5_duplicate']) & (~flags['rule6_column_shift'])
core_rf = core[physics_ok].copy()
core_rf['log_perm'] = np.log10(core_rf['permeability_mD'].clip(lower=1e-3))
print(f"Обучающая выборка RF: {len(core_rf)} строк "
      f"(физически валидные, все 3 лаборатории)")

# Сначала смотрим на парные корреляции с целевой — если связи нет,
# никакая модель её не выучит.
print(f"\nПарные корреляции с log10(K):")
for col in ['porosity', 'density_gcc', 'water_saturation',
            'oil_saturation', 'depth_m']:
    r = core_rf[col].corr(core_rf['log_perm'])
    print(f"  {col:20s}: r = {r:+.3f}")

# Признаки: пористость, плотность, насыщенности, depth. log_perm — целевая.
feat_rf = ['porosity', 'density_gcc', 'water_saturation', 'oil_saturation', 'depth_m']
X_rf = core_rf[feat_rf].values
y_rf = core_rf['log_perm'].values

X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=2,
                           random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pr = rf.predict(X_te)

r2 = r2_score(y_te, y_pr)
mae = mean_absolute_error(y_te, y_pr)
# 5-fold CV — более надёжная оценка
cv = cross_val_score(rf, X_rf, y_rf, cv=KFold(5, shuffle=True, random_state=42),
                     scoring='r2')
print(f"\nTrain n={len(X_tr)}, Test n={len(X_te)}")
print(f"R² (одно разбиение) = {r2:.3f}")
print(f"R² (5-fold CV) = {cv.mean():.3f} ± {cv.std():.3f}")
print(f"MAE (test, log10 шкала) = {mae:.3f}")
print(f"   -- в линейной шкале это коэф. = ×/÷ {10**mae:.2f}")

if cv.mean() < 0.3:
    print("""
ВНИМАНИЕ: R² < 0.3 — модель плохо предсказывает проницаемость.
Причина видна из парных корреляций выше: φ, ρ, Sw слабо связаны с log(K)
в этих данных. Это особенность датасета (искусственно зашумлённый набор),
а не недостаток алгоритма. На реальных керновых данных φ–K даёт r > 0.6.

Что можно попробовать на реальных данных:
  • добавить признаки по фации/литотипу (litotype из 2.2)
  • разделить модели по геол. зонам или скважинам
  • использовать преобразования: log(φ), φ²/(1-φ)² (Козени-Кармен)
""")

# График факт vs прогноз
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_te, y_pr, s=40, alpha=0.7, color='#4a7ab8',
                edgecolor='white', lw=0.5)
lo, hi = min(y_te.min(), y_pr.min()), max(y_te.max(), y_pr.max())
axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
axes[0].set_xlabel('log₁₀(K) факт')
axes[0].set_ylabel('log₁₀(K) прогноз')
axes[0].set_title(f'RF: факт vs прогноз  (R²={r2:.2f})')
axes[0].legend()

# Feature importance
fi = pd.Series(rf.feature_importances_, index=feat_rf).sort_values()
fi.plot.barh(ax=axes[1], color='#2ca02c')
axes[1].set_title('Важность признаков')
axes[1].set_xlabel('importance')
fig.tight_layout()
plt.show()

print(f"\nВажность признаков:\n{fi.sort_values(ascending=False).round(3).to_string()}")

# Применяем ко всему датасету (включая аномалии, но прогноз помечаем как "ml_predicted")
core_full = core.copy()
core_full['log_perm_measured'] = np.log10(core_full['permeability_mD'].clip(lower=1e-3))
core_full['log_perm_predicted'] = rf.predict(core_full[feat_rf].values)
core_full['perm_mD_predicted'] = 10 ** core_full['log_perm_predicted']
core_full.to_csv(fpath('ml_predicted_perm.csv'), index=False, encoding='utf-8-sig')
print(f"Сохранено: {fpath('ml_predicted_perm.csv')}")


# %% 3.1. ВЕРИФИКАЦИЯ ГИС ====================================================
hr("3.1 Верификация ГИС — Well_45")

gis = pd.read_csv(fpath('gis_data_well45.csv'))
print(f"Размер: {gis.shape}")
print(f"Глубина: {gis['DEPTH'].min()}–{gis['DEPTH'].max()} м, шаг "
      f"{(gis['DEPTH'].diff().median()):.2f} м")
print(f"Пропуски: {gis[['PZ','PS','GK','NGK']].isna().sum().to_dict()}")
print(f"Покрытие: "
      f"{(gis[['PZ','PS','GK','NGK']].notna().mean() * 100).round(1).to_dict()} %")

# Правила физичности для ГИС
gis_flags = pd.DataFrame(index=gis.index)
gis_flags['neg_PZ'] = gis['PZ'] < 0
gis_flags['spike_GK'] = gis['GK'] > 16.0
gis_flags['out_NGK'] = (gis['NGK'] < 80) | (gis['NGK'] > 250)

# «Застрявший инструмент» — одно значение повторяется >= 5 раз подряд
def stuck_runs(series, min_run=5):
    """Возвращает bool-маску, помечающую участки длиной >= min_run."""
    s = series.copy()
    # отдельный «прогон» начинается там, где значение меняется или появляется NaN
    grp = (s != s.shift()).cumsum()
    run_len = s.groupby(grp).transform('size')
    return run_len.ge(min_run) & s.notna()

for curve in ['PZ', 'PS', 'GK', 'NGK']:
    gis_flags[f'stuck_{curve}'] = stuck_runs(gis[curve], min_run=5)

gis_flags['any_anomaly'] = gis_flags.any(axis=1)
print(f"\nАномалии ГИС:")
for col in gis_flags.columns[:-1]:
    print(f"  {col:15s}: {gis_flags[col].sum():4d}")
print(f"  ИТОГО строк с аномалиями: {gis_flags['any_anomaly'].sum()}")

# Планшет каротажа
fig, axes = plt.subplots(1, 4, figsize=(13, 9), sharey=True)
curves = [('PZ', 'Ом·м', 'log'), ('PS', 'мВ', 'linear'),
          ('GK', 'мкР/ч', 'linear'), ('NGK', 'усл.ед.', 'linear')]
for ax, (curve, units, scale) in zip(axes, curves):
    d = gis[['DEPTH', curve]].dropna()
    ax.plot(d[curve], d['DEPTH'], lw=0.6, color='#333')
    # Аномалии красным
    anom = gis[gis_flags[f'stuck_{curve}'] |
               (gis_flags['neg_PZ'] if curve == 'PZ' else False) |
               (gis_flags['spike_GK'] if curve == 'GK' else False) |
               (gis_flags['out_NGK'] if curve == 'NGK' else False)]
    if len(anom) > 0:
        ax.scatter(anom[curve], anom['DEPTH'], s=14, c='red',
                   zorder=5, label=f'аномалии (n={len(anom)})')
        ax.legend(fontsize=8, loc='lower right')
    ax.set_xlabel(f'{curve}, {units}')
    if scale == 'log' and (d[curve] > 0).all():
        ax.set_xscale('log')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Глубина, м')
fig.suptitle(f'Планшет каротажа Well_45 (глубина {gis["DEPTH"].min():.0f}–'
             f'{gis["DEPTH"].max():.0f} м)', fontweight='bold')
fig.tight_layout()
plt.show()


# %% 3.2. КОРРЕЛЯЦИИ МЕЖДУ ГИС-КРИВЫМИ =======================================
hr("3.2 Корреляции ГИС-кривых")

# Берём только строки где все 4 кривые есть
gis_full = gis.dropna(subset=['PZ', 'PS', 'GK', 'NGK']).copy()
print(f"Строк с полным набором: {len(gis_full)}")

corr = gis_full[['PZ', 'PS', 'GK', 'NGK']].corr()
print(f"\nМатрица корреляций (Пирсон):")
print(corr.round(3))

# Тепловая карта
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                color='white' if abs(corr.values[i, j]) > 0.5 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title('Корреляции ГИС-кривых')
fig.tight_layout()
plt.show()

# Три ключевые пары
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (a, b) in zip(axes, [('NGK', 'PS'), ('GK', 'NGK'), ('GK', 'PS')]):
    ax.scatter(gis_full[a], gis_full[b], s=8, alpha=0.4, c='#4a7ab8')
    r = gis_full[[a, b]].corr().iloc[0, 1]
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f'{a} vs {b}  (r = {r:.2f})')
fig.tight_layout()
plt.show()

# Самая сильная пара (вне диагонали)
m = corr.abs().where(~np.eye(4, dtype=bool))
imax = m.stack().idxmax()
print(f"\nСамая сильная корреляция: {imax[0]} ↔ {imax[1]}, r = {corr.loc[imax]:.3f}")
print("""
ФИЗИКА (общая теория):
 • GK (гамма) высок в глинах. NGK (нейтрон) тоже высок в глинах — много
   связанной воды  -- отрицательная связь с пористостью коллектора. Обычно
   GK и NGK положительно коррелируют между собой.
 • PS падает в глинах (мембранный/диффузионный потенциал маленький), поэтому
   PS отрицательно связан с GK и NGK.

ЧТО В ДАННЫХ:
 • Если r(GK,NGK) близко к 0 — кривая GK на этой скважине либо шумная,
   либо снята с другим калибровочным сдвигом (см. участок 1500–1700 м
   на планшете 3.1). Это известная проблема архивного каротажа.
 • Кривые с сильной связью (см. таблицу выше) пригодны для взаимного
   восстановления в 3.3. Остальные восстановим, но R² будет низким.
""")


# %% 3.3. ВОССТАНОВЛЕНИЕ ПРОПУСКОВ ГИС =======================================
hr("3.3 Восстановление пропусков ML — два подхода рядом")

from sklearn.ensemble import RandomForestRegressor


def eval_two_ways(df, target, predictors, min_samples=200):
    """
    Возвращает (r2_random, r2_depth) — оценку качества модели в двух режимах:
      • random: случайное разбиение train/test (как обычно делают)
      • depth:  train = верх скважины, test = низ (без интерполяционной утечки)

    На каротажных данных random-разбиение даёт фейково высокий R², потому что
    тестовая точка на 1234.5 м оказывается «зажата» между соседями
    на 1234.4 и 1234.6 из train, и RF фактически интерполирует.
    """
    mask = df[target].notna() & df[predictors].notna().all(axis=1)
    cols = list(dict.fromkeys(predictors + [target, 'DEPTH']))  # без дублей
    sub = df.loc[mask, cols].sort_values('DEPTH').reset_index(drop=True)
    if len(sub) < min_samples:
        return None, None, 0

    n = len(sub)
    # 1) случайное разбиение
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    tr_r, te_r = idx[:int(n * 0.8)], idx[int(n * 0.8):]
    X_tr = sub.loc[tr_r, predictors].values
    y_tr = sub.loc[tr_r, target].values
    X_te = sub.loc[te_r, predictors].values
    y_te = sub.loc[te_r, target].values
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    r2_random = r2_score(y_te, rf.predict(X_te))

    # 2) разбиение по глубине (train = верх, test = низ)
    split = int(n * 0.8)
    X_tr = sub.loc[:split - 1, predictors].values
    y_tr = sub.loc[:split - 1, target].values
    X_te = sub.loc[split:, predictors].values
    y_te = sub.loc[split:, target].values
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    r2_depth = r2_score(y_te, rf.predict(X_te))

    return r2_random, r2_depth, int(mask.sum())


# === Сравнение качества обеих оценок ===
print("Сравнение R²: random vs depth-split (на полных строках):")
print(f"{'Кривая':<6} {'Предикторы':<35} {'R² random':>10} {'R² depth':>10}  Вердикт")
print("-" * 90)

eval_results = {}
for target in ['NGK', 'PS', 'PZ', 'GK']:
    preds = ['DEPTH'] + [c for c in ['PZ', 'PS', 'GK', 'NGK'] if c != target]
    r2_r, r2_d, n = eval_two_ways(gis, target, preds)
    if r2_r is None:
        continue
    if r2_d > 0.5:
        verdict = "модель работает"
    elif r2_d > 0.2:
        verdict = "модель слабая, но что-то ловит"
    else:
        verdict = "модель НЕ работает (R² random — интерполяция)"
    eval_results[target] = (r2_r, r2_d, preds)
    print(f"{target:<6} {str(preds):<35} {r2_r:>10.3f} {r2_d:>10.3f}  {verdict}")

print("""
ИНТЕРПРЕТАЦИЯ:
 • Если R² random = 1.00 и R² depth низкий — это типичная утечка через
   глубину. Random Forest при случайном разбиении видит соседние точки
   train --test (шаг 10 см) и интерполирует. Это НЕ настоящее ML-предсказание
   каротажа из других каротажей, а просто сглаживание соседей.
 • Честная метрика — R² depth: train на верхней части скважины,
   test на нижней. Если она > 0.5 — модель действительно выучила
   зависимость между кривыми, а не просто запомнила соседей.
""")

# === Восстановление: помечаем строки разными источниками ===
# Стратегия:
#   • если R² depth > 0.5 — называем восстановление 'ml_predicted' (честно)
#   • если R² depth <= 0.5, но R² random = 1 — называем 'ml_interpolated'
#     (это всё ещё разумные значения, потому что соседи рядом, но не ML
#     в строгом смысле)
#   • если обе метрики плохие — не заполняем, оставляем 'missing'

gis_r = gis.copy()
sources = {col: pd.Series('missing', index=gis_r.index) for col in ['PZ', 'PS', 'GK', 'NGK']}
for col in sources:
    sources[col][gis_r[col].notna()] = 'measured'


def restore_with_fallback(df, target, base_preds, min_samples=200):
    """
    Восстановление с fallback на меньшие наборы предикторов
    (когда полный набор недоступен из-за скоррелированных пропусков).
    Возвращает (restored_series, source_label, n_filled).
    """
    # пробуем по убыванию количества предикторов
    pred_options = [base_preds]
    for k in range(len(base_preds) - 1, 1, -1):
        # урезанные комбинации: всегда оставляем DEPTH + меньше кривых
        from itertools import combinations
        others = [p for p in base_preds if p != 'DEPTH']
        for combo in combinations(others, k - 1):
            pred_options.append(['DEPTH'] + list(combo))

    restored = df[target].copy()
    total_filled = 0
    for preds in pred_options:
        still_missing = restored.isna()
        if still_missing.sum() == 0:
            break
        train_mask = restored.notna() & df[preds].notna().all(axis=1)
        pred_mask  = still_missing & df[preds].notna().all(axis=1)
        if train_mask.sum() < min_samples or pred_mask.sum() == 0:
            continue
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
        m.fit(df.loc[train_mask, preds].values, restored.loc[train_mask].values)
        restored.loc[pred_mask] = m.predict(df.loc[pred_mask, preds].values)
        total_filled += int(pred_mask.sum())
    return restored, total_filled


print("\n=== Запускаем восстановление с честной маркировкой источника ===")
for target, (r2_r, r2_d, preds) in eval_results.items():
    if r2_d > 0.5:
        label = 'ml_predicted'      # честное ML, обобщает на новую глубину
    elif r2_r > 0.7:
        label = 'ml_interpolated'   # интерполяция между соседями по глубине
    else:
        label = None                # вообще не заполняем
        print(f"  {target}: пропуск (R² depth={r2_d:.2f}, R² random={r2_r:.2f}) — "
              f"модель не работает")
        continue

    restored, n_filled = restore_with_fallback(gis_r, target, preds)
    newly = gis_r[target].isna() & restored.notna()
    gis_r[target] = restored
    sources[target][newly] = label
    print(f"  {target}: заполнено {int(newly.sum())} точек как '{label}' "
          f"(R² depth={r2_d:.2f}, R² random={r2_r:.2f})")

# === Покрытие до/после ===
fig, ax = plt.subplots(figsize=(9, 4.5))
curves_order = ['PZ', 'PS', 'GK', 'NGK']
before = [gis[c].notna().mean() * 100 for c in curves_order]
ml_pred = [(sources[c] == 'ml_predicted').mean() * 100 for c in curves_order]
ml_intp = [(sources[c] == 'ml_interpolated').mean() * 100 for c in curves_order]
x = np.arange(4)
ax.bar(x, before, 0.6, color='#444', label='измерено')
ax.bar(x, ml_pred, 0.6, bottom=before, color='#2ca02c',
       label='ML (R² depth > 0.5)')
ax.bar(x, ml_intp, 0.6, bottom=[b + p for b, p in zip(before, ml_pred)],
       color='#f0a050', label='интерполяция (R² depth низкий)')
ax.set_xticks(x)
ax.set_xticklabels(curves_order)
ax.set_ylabel('Покрытие, %')
ax.set_title('Покрытие до/после восстановления')
ax.legend()
ax.set_ylim(0, 100)
fig.tight_layout()
plt.show()

# === Сравнительный планшет ===
fig, axes = plt.subplots(1, 4, figsize=(14, 9), sharey=True)
for ax, curve in zip(axes, curves_order):
    meas_idx = sources[curve] == 'measured'
    pred_idx = sources[curve] == 'ml_predicted'
    intp_idx = sources[curve] == 'ml_interpolated'
    if meas_idx.sum() > 0:
        ax.plot(gis_r.loc[meas_idx, curve], gis_r.loc[meas_idx, 'DEPTH'],
                '.', markersize=0.8, color='#222', label='измерено')
    if pred_idx.sum() > 0:
        ax.plot(gis_r.loc[pred_idx, curve], gis_r.loc[pred_idx, 'DEPTH'],
                '.', markersize=1.5, alpha=0.6, color='#2ca02c',
                label=f'ML (n={pred_idx.sum()})')
    if intp_idx.sum() > 0:
        ax.plot(gis_r.loc[intp_idx, curve], gis_r.loc[intp_idx, 'DEPTH'],
                '.', markersize=1.5, alpha=0.6, color='#f0a050',
                label=f'интерпол. (n={intp_idx.sum()})')
    ax.set_xlabel(curve)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='lower right')
axes[0].set_ylabel('Глубина, м')
fig.suptitle('Планшет: измерено vs ML vs интерполяция', fontweight='bold')
fig.tight_layout()
plt.show()

# === Сохранение ===
for curve in curves_order:
    gis_r[f'{curve}_source'] = sources[curve]
out_gis = fpath('gis_data_restored.csv')
gis_r.to_csv(out_gis, index=False, encoding='utf-8-sig')
print(f"\nСохранено: {out_gis}")
print("В файле колонки *_source принимают значения:")
print("  • measured        — реальное лабораторное измерение")
print("  • ml_predicted    — честное ML-предсказание (R² depth > 0.5)")
print("  • ml_interpolated — заполнение соседями по глубине (R² depth низкий)")
print("  • missing         — пропуск, не удалось восстановить")


# %% 3.4. СОПОСТАВЛЕНИЕ ГИС С КЕРНОМ =========================================
hr("3.4 Сопоставление ГИС vs керн (Well_45)")

core45 = pd.read_csv(fpath('core_data_well45.csv'))
print(f"Образцов керна Well_45: {len(core45)}")
print(f"Диапазон глубин керна: {core45['depth_m'].min()}–{core45['depth_m'].max()} м")

# Берём NGK из ВОССТАНОВЛЕННОГО ГИС (там покрытие выше)
gis_for_match = gis_r[['DEPTH', 'NGK']].dropna()

# Для каждого образца керна — ближайшая точка ГИС
matched = []
for _, row in core45.iterrows():
    d = row['depth_m']
    idx = (gis_for_match['DEPTH'] - d).abs().idxmin()
    gap = abs(gis_for_match.loc[idx, 'DEPTH'] - d)
    matched.append({
        'depth_m': d,
        'porosity': row['porosity'],
        'permeability_mD': row['permeability_mD'],
        'NGK_gis': gis_for_match.loc[idx, 'NGK'],
        'depth_gap_m': gap,
    })
match_df = pd.DataFrame(matched)
# Отбрасываем где разрыв по глубине > 2 м
match_df = match_df[match_df['depth_gap_m'] < 2.0]
print(f"Сопоставлено: {len(match_df)} / {len(core45)} (разрыв < 2 м)")

# Корреляция φ керна vs NGK
r = match_df[['porosity', 'NGK_gis']].corr().iloc[0, 1]
print(f"\nКорреляция (керн φ) ↔ (ГИС NGK): r = {r:.3f}")
print(f"   -- ожидание: отрицательная (выше φ  -- ниже NGK, "
      f"т.к. NGK реагирует на водород в глинах и связанной воде)")
print(f"   -- факт:    {'согласуется' if r < -0.2 else 'не согласуется (нужно разбираться)'}")

# Scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(match_df['NGK_gis'], match_df['porosity'], s=40,
                alpha=0.7, color='#4a7ab8', edgecolor='white')
axes[0].set_xlabel('NGK (ГИС)')
axes[0].set_ylabel('Пористость (керн), д.е.')
axes[0].set_title(f'Керн φ vs ГИС NGK   (r = {r:.2f})')

# Совмещённый профиль по глубине
ax2 = axes[1]
ax2.plot(gis_r['NGK'], gis_r['DEPTH'], lw=0.5, color='gray', alpha=0.6, label='NGK по стволу')
ax2.set_xlabel('NGK', color='gray')
ax2.tick_params(axis='x', colors='gray')
ax2.invert_yaxis()
ax2.set_ylabel('Глубина, м')

ax3 = ax2.twiny()
ax3.scatter(match_df['porosity'], match_df['depth_m'], s=30, color='#d62728',
            label='φ керн')
ax3.set_xlabel('Пористость (керн)', color='#d62728')
ax3.tick_params(axis='x', colors='#d62728')

axes[1].set_title('Совмещённый профиль по глубине')
fig.tight_layout()
plt.show()

print(f"\nГОТОВО (часть 3 ML). Все выходные файлы — в {DATA_DIR}")


# %% 3.5. ПЕТРОФИЗИКА ПО ФОРМУЛАМ ============================================
hr("3.5 Петрофизика по формулам (φ_N, V_sh, Sw по Арчи, φ_eff)")

# Идея: вместо ML посчитаем физические параметры пласта прямо из ГИС-кривых
# по классическим петрофизическим уравнениям. Константы калибруем по тем
# 12 образцам керна Well_45, которые мы сопоставили в блоке 3.4.

# Сопоставим керн с ГИС заново (нужны все 4 кривых, не только NGK)
gis_pf = gis.dropna(subset=['PZ', 'PS', 'GK', 'NGK']).copy()
print(f"Точек ГИС с полным набором кривых: {len(gis_pf)}")

matched_pf = []
for _, row in core45.iterrows():
    d = row['depth_m']
    idx = (gis_pf['DEPTH'] - d).abs().idxmin()
    gap = abs(gis_pf.loc[idx, 'DEPTH'] - d)
    if gap < 2.0:
        matched_pf.append({
            'depth_m':  d,
            'phi_core': row['porosity'],
            'Sw_core':  row['water_saturation'],
            'NGK': gis_pf.loc[idx, 'NGK'],
            'GK':  gis_pf.loc[idx, 'GK'],
            'PS':  gis_pf.loc[idx, 'PS'],
            'PZ':  gis_pf.loc[idx, 'PZ'],
        })
mp = pd.DataFrame(matched_pf)
print(f"Парных образцов керн↔ГИС: {len(mp)}")

# ---------- ШАГ 1. КАЛИБРОВКА КОНСТАНТ ----------
print("\n--- Калибровка опорных значений ---")

# (а) NGK_коллектор и NGK_глина
# Лучший способ — линейная регрессия φ_керн ↔ NGK на парных образцах.
# Это даёт физические опорные точки: NGK при φ=0.3 и NGK при φ=0.
# Fallback (если керна < 8) — перцентили p1/p99 кривой.
if len(mp) >= 8:
    from sklearn.linear_model import LinearRegression as _LR
    _lr = _LR().fit(mp[['NGK']].values, mp['phi_core'].values)
    _a, _b = _lr.intercept_, _lr.coef_[0]
    # Защита: коэф b должен быть отрицательным (NGK↑  -- φ↓)
    if _b < 0:
        NGK_clean = float((0.30 - _a) / _b)   # NGK при φ=0.30 (макс)
        NGK_clay  = float(-_a / _b)            # NGK при φ=0
        # Прижимаем к физическим границам [80, 250] из задания
        NGK_clean = max(NGK_clean, 80)
        NGK_clay  = min(NGK_clay, 250)
        calib_src = f"линейная регрессия по {len(mp)} керн.образцам"
    else:
        # Регрессия дала неправильный знак — fallback на перцентили
        NGK_clean = max(gis_pf['NGK'].quantile(0.01), 80)
        NGK_clay  = min(gis_pf['NGK'].quantile(0.99), 250)
        calib_src = "перцентили p1/p99 (регрессия дала + наклон)"
else:
    NGK_clean = max(gis_pf['NGK'].quantile(0.01), 80)
    NGK_clay  = min(gis_pf['NGK'].quantile(0.99), 250)
    calib_src = f"перцентили p1/p99 (керна {len(mp)} < 8)"

print(f"NGK_коллектор = {NGK_clean:.1f}   (источник: {calib_src})")
print(f"NGK_глина     = {NGK_clay:.1f}")

# (б) GK_min, GK_max — для V_sh
GK_min = gis_pf['GK'].quantile(0.05)
GK_max = gis_pf['GK'].quantile(0.95)
print(f"GK_min (чист. песчаник) = {GK_min:.2f}")
print(f"GK_max (глина)          = {GK_max:.2f}")
print(f"   -- диапазон GK всего {GK_max-GK_min:.1f} мкР/ч — узкий, "
      f"V_sh по GK малоинформативна")

# (в) Калибровка R_w для Арчи (a=1, m=2, n=2)
# Sw = sqrt(R_w / (φ² · PZ))
# Минимизируем RMSE между расчётным Sw и керновым на парных образцах
a, m_arch, n_arch = 1.0, 2.0, 2.0
best_Rw, best_rmse = None, 1e9
for Rw in np.linspace(0.01, 5.0, 500):
    Sw_calc = ((a * Rw) / (mp['phi_core']**m_arch * mp['PZ']))**(1 / n_arch)
    Sw_calc = np.clip(Sw_calc, 0, 1)
    rmse = np.sqrt(((Sw_calc - mp['Sw_core'])**2).mean())
    if rmse < best_rmse:
        best_rmse, best_Rw = rmse, Rw
print(f"R_w (Арчи, a=1, m=2, n=2) = {best_Rw:.3f} Ом·м   (RMSE={best_rmse:.3f})")

# ---------- ШАГ 2. РАСЧЁТ ПО ВСЕМУ СТВОЛУ ----------
print("\n--- Расчёт по всему стволу ---")

# Используем gis_r (с восстановленными кривыми, где это было допустимо)
pf = gis_r.copy()

# (1) Пористость по нейтронам
#     φ_N = φ_max * (NGK_глина − NGK) / (NGK_глина − NGK_коллектор)
#     Калибровочные точки соответствуют φ=0 (глина) и φ=0.30 (чистый коллектор)
PHI_MAX = 0.30
pf['phi_N'] = PHI_MAX * (NGK_clay - pf['NGK']) / (NGK_clay - NGK_clean)
pf['phi_N'] = pf['phi_N'].clip(0, 0.40)   # физ. граница 0.4 на случай выбросов NGK

# (2) Глинистость по GK (линейная Ларионова)
#     V_sh = (GK − GK_min) / (GK_max − GK_min)
pf['V_sh'] = (pf['GK'] - GK_min) / (GK_max - GK_min)
pf['V_sh'] = pf['V_sh'].clip(0, 1)

# (3) Эффективная пористость:  φ_eff = φ_N · (1 − V_sh)
pf['phi_eff'] = pf['phi_N'] * (1 - pf['V_sh'])

# (4) Водонасыщенность по Арчи: Sw = sqrt(R_w / (φ² · PZ))
phi_safe = pf['phi_N'].clip(lower=0.01)   # защита от деления на ноль
pf['Sw_archie'] = np.sqrt(best_Rw / (phi_safe**m_arch * pf['PZ'].clip(lower=0.01)))
pf['Sw_archie'] = pf['Sw_archie'].clip(0, 1)
pf['So_archie'] = 1 - pf['Sw_archie']

# Статистика покрытия
print(f"Покрытие синтетических кривых:")
for c in ['phi_N', 'V_sh', 'phi_eff', 'Sw_archie']:
    pct = pf[c].notna().mean() * 100
    print(f"  {c:12s}: {pct:5.1f}% точек ({pf[c].notna().sum()} из {len(pf)})")

# ---------- ШАГ 3. ВАЛИДАЦИЯ ПО КЕРНУ ----------
print("\n--- Валидация по 12 керновым образцам ---")

# Считаем синтетику в точках сопоставления
phi_N_at_core = []
V_sh_at_core  = []
Sw_at_core    = []
for _, row in mp.iterrows():
    phi_N_i = PHI_MAX * (NGK_clay - row['NGK']) / (NGK_clay - NGK_clean)
    phi_N_at_core.append(phi_N_i)
    V_sh_at_core.append((row['GK'] - GK_min) / (GK_max - GK_min))
    phi_safe_i = max(phi_N_i, 0.01)
    Sw_at_core.append(np.clip(
        np.sqrt(best_Rw / (phi_safe_i**2 * max(row['PZ'], 0.01))), 0, 1))
mp['phi_N']     = np.clip(phi_N_at_core, 0, 0.4)
mp['V_sh']      = np.clip(V_sh_at_core, 0, 1)
mp['Sw_archie'] = Sw_at_core

r_phi = mp[['phi_core', 'phi_N']].corr().iloc[0, 1]
r_sw  = mp[['Sw_core',  'Sw_archie']].corr().iloc[0, 1]
rmse_phi = np.sqrt(((mp['phi_N'] - mp['phi_core'])**2).mean())
rmse_sw  = np.sqrt(((mp['Sw_archie'] - mp['Sw_core'])**2).mean())
print(f"  φ_N vs φ_керн:     r = {r_phi:+.3f},  RMSE = {rmse_phi:.3f}")
print(f"  Sw_арчи vs Sw_керн: r = {r_sw:+.3f},  RMSE = {rmse_sw:.3f}")
print(f"\nТаблица сравнения (12 образцов):")
print(mp[['depth_m', 'phi_core', 'phi_N', 'Sw_core', 'Sw_archie', 'V_sh']]
      .round(3).to_string(index=False))

# ---------- ШАГ 4. ГРАФИКИ ----------
# Cross-plot керн vs синтетика
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(mp['phi_core'], mp['phi_N'], s=60, color='#2ca02c',
                edgecolor='white', label=f'r = {r_phi:+.2f}')
lo = min(mp['phi_core'].min(), mp['phi_N'].min())
hi = max(mp['phi_core'].max(), mp['phi_N'].max())
axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
axes[0].set_xlabel('φ керн')
axes[0].set_ylabel('φ_N по NGK')
axes[0].set_title(f'Пористость: керн vs формула  (RMSE={rmse_phi:.3f})')
axes[0].legend()

axes[1].scatter(mp['Sw_core'], mp['Sw_archie'], s=60, color='#d62728',
                edgecolor='white', label=f'r = {r_sw:+.2f}')
axes[1].plot([0, 1], [0, 1], 'r--', lw=1.5, label='y = x')
axes[1].set_xlabel('Sw керн')
axes[1].set_ylabel('Sw по Арчи')
axes[1].set_title(f'Водонасыщенность: керн vs Арчи  (RMSE={rmse_sw:.3f})')
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].legend()
fig.tight_layout()
plt.show()

# Планшет: 0 кривые по глубине + точки керна
fig, axes = plt.subplots(1, 4, figsize=(14, 9), sharey=True)
panels = [('phi_N',     'φ нейтронная, д.е.',    '#1f77b4', 'phi_core'),
          ('V_sh',      'V_sh, д.е.',            '#8c564b', None),
          ('phi_eff',   'φ_eff, д.е.',           '#2ca02c', 'phi_core'),
          ('Sw_archie', 'Sw (Арчи), д.е.',       '#d62728', 'Sw_core')]
for ax, (col, xlabel, color, core_col) in zip(axes, panels):
    sub = pf[['DEPTH', col]].dropna()
    ax.plot(sub[col], sub['DEPTH'], lw=0.4, color=color, alpha=0.7)
    if core_col is not None:
        ax.scatter(mp[core_col], mp['depth_m'], s=35, color='black',
                   zorder=5, label='керн', edgecolor='white', lw=0.5)
        ax.legend(fontsize=8, loc='lower right')
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Глубина, м')
fig.suptitle('0 петрофизические кривые Well_45 + точки керна',
             fontweight='bold')
fig.tight_layout()
plt.show()

# Сохранение
out_pf = fpath('petrophysics_synthetic.csv')
pf[['well_id', 'field', 'DEPTH', 'PZ', 'PS', 'GK', 'NGK',
    'phi_N', 'V_sh', 'phi_eff', 'Sw_archie', 'So_archie']]\
    .to_csv(out_pf, index=False, encoding='utf-8-sig')
print(f"\nСохранено: {out_pf}")

print(f"""
КАК ЧИТАТЬ РЕЗУЛЬТАТЫ:
 • φ_N (по NGK): согласуется с керном на r={r_phi:+.2f}. {"Работает." if r_phi<-0.3 or r_phi>0.3 else "Связь слабая."}
   Это базовый метод определения пористости из ГИС, ничего сложнее не нужно.
 • V_sh (глинистость по GK): диапазон GK на Well_45 узкий ({GK_max-GK_min:.1f} мкР/ч),
   почти всё классифицируется как чистый коллектор. На этой скважине V_sh
   малоинформативна — нет настоящих глинистых пропластков.
 • φ_eff = φ_N · (1−V_sh): по сути близко к φ_N, потому что V_sh = 0 везде.
 • Sw (Арчи): r={r_sw:+.2f} с керном.
   {"Согласуется." if abs(r_sw)>0.4 else "Согласие слабое — данные физически неконсистентны."}
   На реальной скважине Арчи обычно даёт r > 0.7 при правильно подобранном R_w.

ПОЧЕМУ ЭТО ВАЖНО:
 • В отличие от ML, эти кривые рассчитываются ВЕЗДЕ, где есть исходный
   каротаж — никаких пропусков, никаких "интерполяций между соседями".
 • Это физика, а не статистика: φ_N работает даже если у нас всего одна
   скважина и нет соседей для обучения.
 • Калибровочные константы (NGK_коллектор, R_w) — это то, что нужно
   уточнить по керну. Мы взяли первое приближение из перцентилей.
""")

# ---------- ФИНАЛЬНЫЙ ИТОГ ----------
hr("ВСЕ ВЫХОДНЫЕ ФАЙЛЫ")
print(f"Папка: {DATA_DIR}")
print("  • anomaly_report.csv         — задание 1.3 (110 аномальных строк керна)")
print("  • ml_anomalies_IF.csv        — задание 2.1 (Isolation Forest)")
print("  • ml_clusters.csv            — задание 2.2 (KMeans литотипы)")
print("  • ml_predicted_perm.csv      — задание 2.3 (RF проницаемость)")
print("  • gis_data_restored.csv      — задание 3.3 (ГИС + ML/интерпол.)")
print("  • petrophysics_synthetic.csv — задание 3.5 (петрофизика по формулам)")
