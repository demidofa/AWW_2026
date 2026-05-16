# -*- coding: utf-8 -*-
"""
Well Data Verifier — PyQt6 приложение для верификации керновых данных
и пространственного восстановления свойств скважин.

Запуск:
    python well_verifier.py

Требования:
    pip install PyQt6 pandas numpy matplotlib scipy scikit-learn

Логика разделена:
    • core.py  — вычислительные функции (verify, restore, …) — не импортирует Qt
    • this    — только UI

Структура окна:
    Шапка: 2 кнопки — «Загрузить архив» и «Загрузить скважину» + лейбл статуса
    6 вкладок:
        1. Сводка        — метрики + лабы + таблица аномалий
        2. Карты         — 3 карты скважин (зоны / пористость / ошибки)
        3. Профили       — пористость по глубине
        4. Вариограмма   — пространственный анализ
        5. Восстановление одной скважины — после «Загрузить скважину»
        6. Прогноз из окружения          — dropdown + ввод координат вручную
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableView, QTextEdit, QComboBox,
    QLineEdit, QGroupBox, QSplitter, QFrame, QSizePolicy, QHeaderView,
)
from PyQt6.QtCore import QAbstractTableModel, QModelIndex

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# =============================================================================
#                          ВЫЧИСЛИТЕЛЬНОЕ ЯДРО
# =============================================================================
# Все функции принимают и возвращают DataFrame — не зависят от Qt.

COLUMN_ALIASES = {
    'well_id':          ['well_id', 'well', 'скважина'],
    'x_m':              ['x_m', 'x'],
    'y_m':              ['y_m', 'y'],
    'zone':             ['zone', 'зона', 'геол_зона'],
    'data_type':        ['data_type', 'type', 'тип'],
    'depth_m':          ['depth_m', 'глубина', 'depth'],
    'porosity':         ['porosity', 'пористость', 'phi'],
    'permeability_mD':  ['permeability_mD', 'permeability', 'perm_md', 'perm',
                          'проницаемость', 'k_md'],
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
    new = {}
    for c in df.columns:
        key = str(c).lower().strip().lstrip('\ufeff')
        new[c] = ALIAS_TO_CANON.get(key, c)
    return df.rename(columns=new)


def read_csv_smart(path):
    """Читает CSV в любом из форматов архива."""
    candidates = []
    for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
        for sep in [',', ';']:
            for dec in ['.', ',']:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc,
                                      decimal=dec, low_memory=False)
                    if df.shape[1] < 3:
                        continue
                    n_num = sum(1 for c in df.columns
                                 if pd.api.types.is_numeric_dtype(df[c]))
                    candidates.append((n_num, df))
                except Exception:
                    continue
    if not candidates:
        raise RuntimeError(f"Не удалось прочитать {path}")
    candidates.sort(key=lambda t: -t[0])
    return normalize_columns(candidates[0][1])


def load_archive(archive_dir, progress_callback=None):
    """Читает реестр + все файлы скважин. Возвращает (core_df, gis_df, register).

    Если задан progress_callback(i, total, message), вызывается после
    каждого файла для обновления прогресса.
    """
    register = pd.read_csv(os.path.join(archive_dir, 'wells_register.csv'))
    files = sorted(glob.glob(os.path.join(archive_dir, 'Well_*.csv')))

    dfs = []
    total = len(files)
    for i, path in enumerate(files):
        if progress_callback:
            progress_callback(i + 1, total, os.path.basename(path))
        df = read_csv_smart(path)
        if 'data_type' in df.columns:
            df['data_type'] = df['data_type'].astype(str).str.lower().str.strip()
        dfs.append(df)
    if progress_callback:
        progress_callback(total, total, "объединение данных…")
    master = pd.concat(dfs, ignore_index=True)

    # Числовые колонки
    for c in ['x_m', 'y_m', 'depth_m', 'porosity', 'permeability_mD',
              'density_gcc', 'water_saturation', 'oil_saturation', 'year',
              'PZ', 'PS', 'GK', 'NGK']:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors='coerce')

    core_cols = ['well_id', 'x_m', 'y_m', 'zone', 'depth_m', 'porosity',
                 'permeability_mD', 'density_gcc', 'water_saturation',
                 'oil_saturation', 'lab_id', 'year']
    gis_cols = ['well_id', 'x_m', 'y_m', 'zone', 'depth_m',
                'PZ', 'PS', 'GK', 'NGK']

    if 'data_type' in master.columns:
        core_df = master[master['data_type'] == 'core'][
            [c for c in core_cols if c in master.columns]].copy()
        gis_df = master[master['data_type'] == 'gis'][
            [c for c in gis_cols if c in master.columns]].copy()
    else:
        # Без data_type: считаем что всё это керн
        core_df = master[[c for c in core_cols if c in master.columns]].copy()
        gis_df = master.iloc[0:0]

    if 'depth_m' in gis_df.columns:
        gis_df = gis_df.rename(columns={'depth_m': 'DEPTH'})
    if 'zone' in core_df.columns:
        core_df['геол_зона'] = core_df['zone']

    return core_df, gis_df, register


def load_single_well(path):
    """Загружает один файл скважины (из архива или совместимого формата)."""
    df = read_csv_smart(path)
    if 'data_type' in df.columns:
        df['data_type'] = df['data_type'].astype(str).str.lower().str.strip()
    for c in ['x_m', 'y_m', 'depth_m', 'porosity', 'permeability_mD',
              'density_gcc', 'water_saturation', 'oil_saturation', 'year',
              'PZ', 'PS', 'GK', 'NGK']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    core_cols = ['well_id', 'x_m', 'y_m', 'zone', 'depth_m', 'porosity',
                 'permeability_mD', 'density_gcc', 'water_saturation',
                 'oil_saturation', 'lab_id', 'year']
    if 'data_type' in df.columns:
        core_df = df[df['data_type'] == 'core'][
            [c for c in core_cols if c in df.columns]].copy()
    else:
        core_df = df[[c for c in core_cols if c in df.columns]].copy()
    if 'zone' in core_df.columns:
        core_df['геол_зона'] = core_df['zone']
    return core_df


def verify_core(df_core):
    """Возвращает df с дополнительными колонками-флагами + словарь счётчиков."""
    df = df_core.copy().reset_index(drop=True)
    flags = pd.DataFrame(index=df.index)
    flags['ошибка_пористость']    = ~df['porosity'].between(0, 0.4)
    flags['ошибка_проницаемость'] = df['permeability_mD'] < 0
    flags['ошибка_плотность']     = ~df['density_gcc'].between(2.0, 3.0)
    sat_sum = df['water_saturation'].fillna(0) + df['oil_saturation'].fillna(0)
    flags['ошибка_насыщенность']  = sat_sum > 1.001
    key = ['well_id', 'depth_m', 'porosity', 'permeability_mD', 'density_gcc']
    key_present = [c for c in key if c in df.columns]
    flags['дубль'] = df.duplicated(subset=key_present, keep=False)
    flags['смещение_столбца'] = df['density_gcc'] > 10

    df_out = pd.concat([df, flags], axis=1)
    flag_cols = list(flags.columns)
    df_out['список_ошибок'] = df_out[flag_cols].apply(
        lambda r: '; '.join([c for c in flag_cols if r[c]]), axis=1)
    df_out['есть_ошибка'] = df_out['список_ошибок'] != ''

    counts = {c: int(df_out[c].sum()) for c in flag_cols}
    counts['всего'] = len(df_out)
    counts['аномалий'] = int(df_out['есть_ошибка'].sum())
    counts['чистых'] = counts['всего'] - counts['аномалий']
    return df_out, counts


def lab_kruskal(df_clean):
    """Возвращает (H, p, dict лаб → bias%)."""
    labs = sorted(df_clean['lab_id'].dropna().unique())
    groups = [df_clean[df_clean['lab_id'] == lab]['porosity'].dropna().values
              for lab in labs]
    if len(labs) < 2 or any(len(g) < 5 for g in groups):
        return None, None, {}, labs
    H, p = stats.kruskal(*groups)
    overall_med = df_clean['porosity'].median()
    bias = {lab: (df_clean[df_clean['lab_id'] == lab]['porosity'].median()
                  - overall_med) / overall_med * 100
            for lab in labs}
    return H, p, bias, labs


def predict_well_from_neighbors(core_clean, x, y):
    """Предсказание медианных свойств скважины по (x, y) через RF.

    Учим на медианах остальных скважин (Lab_A + Lab_B), предсказываем целевую.
    Возвращает {phi, K, density, std_phi, std_K}.
    """
    sub = core_clean[core_clean['lab_id'].isin(['Lab_A', 'Lab_B'])]
    ws = sub.groupby('well_id').agg(
        x=('x_m', 'first'),
        y=('y_m', 'first'),
        phi_med=('porosity', 'median'),
        K_med=('permeability_mD', 'median'),
        rho_med=('density_gcc', 'median'),
    ).reset_index()
    if len(ws) < 5:
        return None

    X_tr = ws[['x', 'y']].values

    rf_phi = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
    rf_phi.fit(X_tr, ws['phi_med'].values)
    phi_pred = float(rf_phi.predict([[x, y]])[0])
    phi_std = float(np.std([t.predict([[x, y]])[0] for t in rf_phi.estimators_]))

    rf_K = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    rf_K.fit(X_tr, np.log10(ws['K_med'].clip(lower=1e-3)).values)
    logK_pred = float(rf_K.predict([[x, y]])[0])
    logK_std = float(np.std([t.predict([[x, y]])[0] for t in rf_K.estimators_]))

    rf_rho = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
    rf_rho.fit(X_tr, ws['rho_med'].values)
    rho_pred = float(rf_rho.predict([[x, y]])[0])
    rho_std = float(np.std([t.predict([[x, y]])[0] for t in rf_rho.estimators_]))

    return {
        'phi':     phi_pred, 'phi_std': phi_std,
        'K':       10 ** logK_pred,
        'K_low':   10 ** (logK_pred - logK_std),
        'K_high':  10 ** (logK_pred + logK_std),
        'rho':     rho_pred, 'rho_std': rho_std,
        'n_neighbors': len(ws),
    }


def loowo_cv(core_clean):
    """Leave-One-Well-Out по медианам — оценка качества прогноза."""
    sub = core_clean[core_clean['lab_id'].isin(['Lab_A', 'Lab_B'])]
    ws = sub.groupby('well_id').agg(
        x=('x_m', 'first'),
        y=('y_m', 'first'),
        phi_med=('porosity', 'median'),
        K_med=('permeability_mD', 'median'),
        rho_med=('density_gcc', 'median'),
    ).reset_index()
    if len(ws) < 5:
        return {}
    preds_phi, true_phi = [], []
    preds_K, true_K = [], []
    preds_rho, true_rho = [], []
    for i in range(len(ws)):
        train = ws.drop(i)
        X_tr = train[['x', 'y']].values
        X_te = [[ws.iloc[i]['x'], ws.iloc[i]['y']]]

        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_tr, train['phi_med'].values)
        preds_phi.append(rf.predict(X_te)[0])
        true_phi.append(ws.iloc[i]['phi_med'])

        rf.fit(X_tr, np.log10(train['K_med'].clip(lower=1e-3)).values)
        preds_K.append(rf.predict(X_te)[0])
        true_K.append(np.log10(max(ws.iloc[i]['K_med'], 1e-3)))

        rf.fit(X_tr, train['rho_med'].values)
        preds_rho.append(rf.predict(X_te)[0])
        true_rho.append(ws.iloc[i]['rho_med'])

    return {
        'r2_phi': r2_score(true_phi, preds_phi),
        'r2_K':   r2_score(true_K, preds_K),
        'r2_rho': r2_score(true_rho, preds_rho),
        'n':      len(ws),
    }


# =============================================================================
#                              QT МОДЕЛЬ ТАБЛИЦЫ
# =============================================================================

class PandasModel(QAbstractTableModel):
    """Модель для отображения DataFrame в QTableView."""
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df.reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()):
        return self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            v = self._df.iat[index.row(), index.column()]
            if pd.isna(v):
                return ""
            if isinstance(v, float):
                return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
            return str(v)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            v = self._df.iat[index.row(), index.column()]
            if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])

    def update(self, df):
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()


# =============================================================================
#                       MATPLOTLIB ВИДЖЕТ ДЛЯ PYQT
# =============================================================================

class MplCanvas(FigureCanvas):
    """Холст matplotlib для встраивания в Qt."""
    def __init__(self, parent=None, figsize=(7, 5)):
        self.fig = Figure(figsize=figsize, facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)

    def clear(self):
        self.fig.clear()
        self.draw()


# =============================================================================
#                              СТИЛЬ (CSS-Qt)
# =============================================================================

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #ffffff;
    color: #1a1a1a;
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #e5e5e5;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: #ffffff;
    color: #666;
    border: none;
    padding: 11px 22px;
    margin-right: 2px;
    font-size: 13px;
    font-weight: 500;
}
QTabBar::tab:hover {
    color: #1a1a1a;
}
QTabBar::tab:selected {
    color: #1a1a1a;
    border-bottom: 2px solid #1a1a1a;
}
QPushButton {
    background-color: #1a1a1a;
    color: #ffffff;
    border: none;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 500;
    border-radius: 2px;
}
QPushButton:hover { background-color: #333; }
QPushButton:pressed { background-color: #000; }
QPushButton:disabled { background-color: #ccc; color: #888; }
QPushButton#secondary {
    background-color: #ffffff;
    color: #1a1a1a;
    border: 1px solid #1a1a1a;
}
QPushButton#secondary:hover {
    background-color: #f5f5f5;
}
QLabel#title {
    font-size: 18px;
    font-weight: 600;
    color: #1a1a1a;
    padding: 8px 0;
}
QLabel#subtitle {
    color: #666;
    font-size: 12px;
}
QLabel#metric-value {
    font-size: 22px;
    font-weight: 600;
    color: #1a1a1a;
}
QLabel#metric-label {
    color: #888;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox {
    border: 1px solid #e5e5e5;
    border-radius: 2px;
    margin-top: 14px;
    padding: 14px;
    font-weight: 600;
    color: #1a1a1a;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px 0 6px;
    background-color: #ffffff;
}
QLineEdit, QComboBox {
    border: 1px solid #d4d4d4;
    border-radius: 2px;
    padding: 7px 10px;
    background: #ffffff;
    font-size: 13px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #1a1a1a;
}
QTableView {
    background: #ffffff;
    alternate-background-color: #fafafa;
    gridline-color: #f0f0f0;
    border: 1px solid #e5e5e5;
    selection-background-color: #1a1a1a;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #f5f5f5;
    color: #1a1a1a;
    padding: 8px 6px;
    border: none;
    border-right: 1px solid #e5e5e5;
    border-bottom: 1px solid #e5e5e5;
    font-weight: 600;
    font-size: 12px;
}
QTextEdit {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 2px;
    padding: 10px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
}
QFrame#hr {
    background-color: #e5e5e5;
    max-height: 1px;
}
QFrame#card {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 2px;
}
"""


# =============================================================================
#                            ВИДЖЕТЫ ВКЛАДОК
# =============================================================================

def make_metric_card(label_text, value_text="—"):
    """Карточка с метрикой: маленький лейбл + большое число."""
    frame = QFrame()
    frame.setObjectName('card')
    frame.setMinimumWidth(140)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(6)
    lbl = QLabel(label_text)
    lbl.setObjectName('metric-label')
    val = QLabel(value_text)
    val.setObjectName('metric-value')
    layout.addWidget(lbl)
    layout.addWidget(val)
    return frame, val


class SummaryTab(QWidget):
    """Вкладка 1: метрики + лабы + таблица аномалий."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Метрики ряд
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self.cards = {}
        for key, label in [
            ('total',     'ВСЕГО ОБРАЗЦОВ'),
            ('clean',     'ЧИСТЫХ'),
            ('anomalies', 'АНОМАЛИЙ'),
            ('wells',     'СКВАЖИН'),
        ]:
            frame, val = make_metric_card(label, "—")
            self.cards[key] = val
            metrics_row.addWidget(frame)
        metrics_row.addStretch()
        layout.addLayout(metrics_row)

        # Сплиттер: график + таблица
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая часть — лабы
        left = QGroupBox("Сравнение лабораторий")
        left_lay = QVBoxLayout(left)
        self.lab_canvas = MplCanvas(figsize=(6, 4))
        left_lay.addWidget(self.lab_canvas)
        self.lab_text = QLabel("—")
        self.lab_text.setWordWrap(True)
        left_lay.addWidget(self.lab_text)
        splitter.addWidget(left)

        # Правая — таблица аномалий
        right = QGroupBox("Найденные аномалии")
        right_lay = QVBoxLayout(right)
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table_model = PandasModel()
        self.table.setModel(self.table_model)
        right_lay.addWidget(self.table)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

    def update_data(self, core_with_flags, counts):
        # Метрики
        self.cards['total'].setText(str(counts.get('всего', '—')))
        self.cards['clean'].setText(str(counts.get('чистых', '—')))
        self.cards['anomalies'].setText(str(counts.get('аномалий', '—')))
        self.cards['wells'].setText(str(core_with_flags['well_id'].nunique()))

        # График лаб
        self.lab_canvas.clear()
        ax = self.lab_canvas.fig.add_subplot(111)
        labs = sorted(core_with_flags['lab_id'].dropna().unique())
        if labs:
            data = [core_with_flags[core_with_flags['lab_id'] == lab]['porosity']
                                                     .dropna().values
                    for lab in labs]
            bp = ax.boxplot(data, tick_labels=labs, patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5),
                            widths=0.55)
            colors = ['#3b82f6', '#10b981', '#ef4444']
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
                patch.set_edgecolor('none')
            ax.set_ylabel('Пористость, д.е.', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=10)
            ax.grid(True, axis='y', alpha=0.25, linewidth=0.6)
        self.lab_canvas.fig.tight_layout()
        self.lab_canvas.draw()

        # Текст про лабы — Крускал-Уоллис
        clean_for_test = core_with_flags[~core_with_flags['есть_ошибка']]
        H, p, bias, _labs = lab_kruskal(clean_for_test)
        if H is not None:
            sig = ('значимо различаются' if p < 0.05
                   else 'различия не значимы')
            bias_str = ', '.join(f"{k}: {v:+.1f}%" for k, v in bias.items())
            self.lab_text.setText(
                f"Kruskal-Wallis: H={H:.2f}, p={p:.2e} — {sig}\n"
                f"Отклонение медиан от общей: {bias_str}")
        else:
            self.lab_text.setText("Недостаточно данных для статистического теста.")

        # Таблица аномалий
        anom = core_with_flags[core_with_flags['есть_ошибка']].copy()
        show_cols = ['well_id', 'depth_m', 'porosity', 'permeability_mD',
                     'density_gcc', 'water_saturation', 'oil_saturation',
                     'lab_id', 'список_ошибок']
        show_cols = [c for c in show_cols if c in anom.columns]
        self.table_model.update(anom[show_cols] if len(anom) > 0
                                 else pd.DataFrame({'info': ['Аномалий не найдено']}))


class MapsTab(QWidget):
    """Вкладка 2: 3 карты скважин."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        self.canvas = MplCanvas(figsize=(16, 7))
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas, 1)

    def update_data(self, core_with_flags, register):
        self.canvas.clear()
        axes = self.canvas.fig.subplots(1, 3)

        zone_colors = {'Зона_А': '#3b82f6', 'Зона_Б': '#10b981', 'Зона_В': '#ef4444'}
        clean = core_with_flags[~core_with_flags['есть_ошибка']]
        ws = clean.groupby('well_id').agg(
            x=('x_m', 'first'), y=('y_m', 'first'),
            phi_med=('porosity', 'median'),
            n=('porosity', 'count'),
        ).reset_index()
        ws['zone'] = ws['well_id'].map(
            core_with_flags[['well_id', 'геол_зона']].drop_duplicates()
                                                       .set_index('well_id')['геол_зона'])

        # 1. Зоны
        ax = axes[0]
        for zone, grp in ws.groupby('zone'):
            ax.scatter(grp['x'], grp['y'], c=zone_colors.get(zone, '#888'),
                       s=grp['n'] * 4, label=zone,
                       edgecolors='white', linewidth=1, zorder=3)
        if register is not None:
            no_data = register[~register['file_in_archive']]
            ax.scatter(no_data['x_m'], no_data['y_m'],
                       c='#e0e0e0', s=40, marker='s',
                       edgecolors='#aaa', linewidth=0.8,
                       label='нет данных', zorder=2)
        ax.set_title('Карта скважин', fontsize=11, color='#1a1a1a')
        ax.set_xlabel('X, м', fontsize=9)
        ax.set_ylabel('Y, м', fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # 2. Пористость
        ax = axes[1]
        sc = ax.scatter(ws['x'], ws['y'], c=ws['phi_med'],
                         cmap='RdYlGn', s=120,
                         edgecolors='white', linewidth=1, vmin=0.05, vmax=0.18)
        self.canvas.fig.colorbar(sc, ax=ax, fraction=0.04,
                                   label='Медиана φ')
        ax.set_title('Медианная пористость', fontsize=11, color='#1a1a1a')
        ax.set_xlabel('X, м', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # 3. Ошибки
        ax = axes[2]
        anom = core_with_flags[core_with_flags['есть_ошибка']]
        if len(anom) > 0:
            anom_count = anom.groupby('well_id').size().reset_index(name='n_err')
            anom_count['x'] = anom_count['well_id'].map(
                core_with_flags.drop_duplicates('well_id')
                                .set_index('well_id')['x_m'])
            anom_count['y'] = anom_count['well_id'].map(
                core_with_flags.drop_duplicates('well_id')
                                .set_index('well_id')['y_m'])
            ax.scatter(ws['x'], ws['y'], c='#e0e0e0', s=40,
                       edgecolors='#aaa', linewidth=0.5, zorder=2)
            sc = ax.scatter(anom_count['x'], anom_count['y'],
                             c=anom_count['n_err'], cmap='Reds',
                             s=anom_count['n_err'] * 60,
                             edgecolors='#7a0000', linewidth=1, zorder=3)
            self.canvas.fig.colorbar(sc, ax=ax, fraction=0.04,
                                       label='Кол-во ошибок')
        else:
            ax.scatter(ws['x'], ws['y'], c='#e0e0e0', s=40,
                       edgecolors='#aaa', linewidth=0.5)
            ax.text(0.5, 0.95, 'Аномалий не найдено',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=10, color='#666')
        ax.set_title('Распределение ошибок', fontsize=11, color='#1a1a1a')
        ax.set_xlabel('X, м', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        self.canvas.fig.tight_layout()
        self.canvas.draw()


class ProfilesTab(QWidget):
    """Вкладка 3: профили пористости по глубине у соседей."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Выбор пары
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Скважина 1:"))
        self.well1 = QComboBox()
        ctrl.addWidget(self.well1)
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("Скважина 2:"))
        self.well2 = QComboBox()
        ctrl.addWidget(self.well2)
        ctrl.addStretch()
        self.btn = QPushButton("Сравнить")
        self.btn.setObjectName('secondary')
        ctrl.addWidget(self.btn)
        layout.addLayout(ctrl)

        self.canvas = MplCanvas(figsize=(10, 5))
        layout.addWidget(self.canvas)

        self._core = None
        self.btn.clicked.connect(self._plot)

    def update_data(self, core_with_flags):
        self._core = core_with_flags[~core_with_flags['есть_ошибка']]
        wells = sorted(self._core['well_id'].unique())
        self.well1.clear(); self.well2.clear()
        self.well1.addItems(wells)
        self.well2.addItems(wells)
        if len(wells) > 1:
            self.well2.setCurrentIndex(1)
        self._plot()

    def _plot(self):
        if self._core is None:
            return
        w1 = self.well1.currentText()
        w2 = self.well2.currentText()
        if not w1 or not w2:
            return
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        colors = ['#3b82f6', '#ef4444']
        for well, color in zip([w1, w2], colors):
            d = self._core[self._core['well_id'] == well].sort_values('depth_m')
            ax.scatter(d['porosity'], d['depth_m'], c=color, alpha=0.45,
                       s=22, label=well, edgecolor='none')
            if len(d) > 5:
                sm = uniform_filter1d(d['porosity'].values, size=5)
                ax.plot(sm, d['depth_m'].values, color=color, lw=1.8,
                        alpha=0.85)
        # Расстояние между скважинами
        try:
            x1 = self._core[self._core['well_id']==w1]['x_m'].iloc[0]
            y1 = self._core[self._core['well_id']==w1]['y_m'].iloc[0]
            x2 = self._core[self._core['well_id']==w2]['x_m'].iloc[0]
            y2 = self._core[self._core['well_id']==w2]['y_m'].iloc[0]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            ax.set_title(f'{w1} vs {w2}  (расстояние {dist:.0f} м)',
                          fontsize=11, color='#1a1a1a')
        except Exception:
            ax.set_title(f'{w1} vs {w2}', fontsize=11)
        ax.set_xlabel('Пористость, д.е.', fontsize=10)
        ax.set_ylabel('Глубина, м', fontsize=10)
        ax.invert_yaxis()
        ax.legend(fontsize=10, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class VariogramTab(QWidget):
    """Вкладка 4: вариограмма."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        self.canvas = MplCanvas(figsize=(11, 5))
        layout.addWidget(self.canvas)
        self.info = QLabel("—")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

    def update_data(self, core_with_flags):
        clean = core_with_flags[~core_with_flags['есть_ошибка']]
        ws = clean.groupby('well_id').agg(
            x=('x_m', 'first'), y=('y_m', 'first'),
            phi=('porosity', 'median'),
        ).reset_index()
        pairs = []
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                r1, r2 = ws.iloc[i], ws.iloc[j]
                dist = np.sqrt((r1['x']-r2['x'])**2 + (r1['y']-r2['y'])**2)
                gamma = 0.5 * (r1['phi'] - r2['phi'])**2
                pairs.append({'dist': dist, 'gamma': gamma,
                              'diff': abs(r1['phi'] - r2['phi'])})
        pdf = pd.DataFrame(pairs)
        if len(pdf) < 5:
            self.canvas.clear()
            self.info.setText("Недостаточно скважин для вариограммы.")
            return

        self.canvas.clear()
        axes = self.canvas.fig.subplots(1, 2)

        # Вариограмма
        ax = axes[0]
        ax.scatter(pdf['dist'], pdf['gamma'], s=12, alpha=0.35,
                   color='#3b82f6', edgecolor='none')
        # Биннинг
        bins = np.linspace(0, pdf['dist'].max(), 12)
        centers = (bins[:-1] + bins[1:]) / 2
        means = []
        for k in range(len(bins) - 1):
            m = (pdf['dist'] >= bins[k]) & (pdf['dist'] < bins[k + 1])
            means.append(pdf.loc[m, 'gamma'].mean() if m.sum() > 3 else np.nan)
        ax.plot(centers, means, 'o-', color='#1a1a1a', lw=1.8, ms=7,
                label='среднее по бинам')
        ax.set_xlabel('Расстояние, м', fontsize=10)
        ax.set_ylabel('γ(h) = 0.5·(φ₁−φ₂)²', fontsize=10)
        ax.set_title('Вариограмма пористости', fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.25, linewidth=0.6)

        # Сходство vs расстояние
        ax = axes[1]
        ax.scatter(pdf['dist'], pdf['diff'], s=12, alpha=0.35,
                   color='#10b981', edgecolor='none')
        z = np.polyfit(pdf['dist'], pdf['diff'], 1)
        xl = np.linspace(pdf['dist'].min(), pdf['dist'].max(), 100)
        ax.plot(xl, np.polyval(z, xl), 'r--', lw=1.5, label='линейный тренд')
        ax.set_xlabel('Расстояние, м', fontsize=10)
        ax.set_ylabel('|φ₁ − φ₂|', fontsize=10)
        ax.set_title('Сходство пористости vs расстояние', fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.25, linewidth=0.6)

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=9)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # Текстовая сводка
        r_diff_dist = pdf[['dist', 'diff']].corr().iloc[0, 1]
        self.info.setText(
            f"Корреляция |Δφ| vs расстояние: r = {r_diff_dist:+.3f}  "
            f"({'есть пространственная связь' if r_diff_dist > 0.15 else 'связь слабая'})")


class SingleWellTab(QWidget):
    """Вкладка 5: после Загрузить скважину — верификация + прогноз по соседям."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        self.title = QLabel("Загрузите скважину для анализа.")
        self.title.setObjectName('title')
        layout.addWidget(self.title)

        # Карточки
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self.cards = {}
        for k, lbl in [('n', 'ОБРАЗЦОВ'),
                       ('errors', 'ОШИБОК'),
                       ('phi', 'МЕДИАНА φ'),
                       ('K', 'МЕДИАНА K, мД')]:
            f, v = make_metric_card(lbl, "—")
            self.cards[k] = v
            cards_row.addWidget(f)
        cards_row.addStretch()
        layout.addLayout(cards_row)

        # Сплиттер: график + сравнение с прогнозом
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QGroupBox("Профиль пористости по глубине")
        ll = QVBoxLayout(left)
        self.profile_canvas = MplCanvas(figsize=(6, 5))
        ll.addWidget(self.profile_canvas)
        splitter.addWidget(left)

        right = QGroupBox("Сравнение с прогнозом по соседям")
        rl = QVBoxLayout(right)
        self.compare_text = QTextEdit()
        self.compare_text.setReadOnly(True)
        rl.addWidget(self.compare_text)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

    def update_data(self, well_df, archive_core_clean=None):
        if well_df is None or len(well_df) == 0:
            self.title.setText("Скважина не загружена.")
            return

        well_id = well_df['well_id'].iloc[0] if 'well_id' in well_df.columns else '—'
        self.title.setText(f"Скважина: {well_id}")

        # Верификация
        v, counts = verify_core(well_df)

        self.cards['n'].setText(str(len(well_df)))
        self.cards['errors'].setText(str(counts['аномалий']))
        phi_med = v[~v['есть_ошибка']]['porosity'].median() \
                   if (~v['есть_ошибка']).any() else np.nan
        K_med = v[~v['есть_ошибка']]['permeability_mD'].median() \
                 if (~v['есть_ошибка']).any() else np.nan
        self.cards['phi'].setText(f"{phi_med:.3f}" if not pd.isna(phi_med) else "—")
        self.cards['K'].setText(f"{K_med:.1f}" if not pd.isna(K_med) else "—")

        # Профиль
        self.profile_canvas.clear()
        ax = self.profile_canvas.fig.add_subplot(111)
        clean = v[~v['есть_ошибка']]
        anom = v[v['есть_ошибка']]
        if len(clean) > 0:
            ax.scatter(clean['porosity'], clean['depth_m'], s=25, alpha=0.6,
                       color='#3b82f6', edgecolor='none', label='чистые')
            if len(clean) > 5:
                d_sorted = clean.sort_values('depth_m')
                sm = uniform_filter1d(d_sorted['porosity'].values, size=5)
                ax.plot(sm, d_sorted['depth_m'].values, color='#1a1a1a',
                        lw=1.5, alpha=0.85)
        if len(anom) > 0:
            ax.scatter(anom['porosity'], anom['depth_m'], s=45, color='#ef4444',
                       marker='x', label='аномалии', linewidths=2)
        ax.set_xlabel('Пористость, д.е.', fontsize=10)
        ax.set_ylabel('Глубина, м', fontsize=10)
        ax.invert_yaxis()
        ax.legend(fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        self.profile_canvas.fig.tight_layout()
        self.profile_canvas.draw()

        # Сравнение с прогнозом по соседям
        if archive_core_clean is not None and 'x_m' in well_df.columns:
            try:
                x = float(well_df['x_m'].iloc[0])
                y = float(well_df['y_m'].iloc[0])
                pred = predict_well_from_neighbors(archive_core_clean, x, y)
                if pred is not None:
                    # Истина по этой скважине
                    phi_true = phi_med
                    K_true = K_med
                    rho_true = v[~v['есть_ошибка']]['density_gcc'].median()
                    text = f"""Координаты: x = {x:.0f}, y = {y:.0f}
Соседних скважин для прогноза: {pred['n_neighbors']}

ПАРАМЕТР               ФАКТ      ПРОГНОЗ    ОШИБКА
Пористость, д.е.    {phi_true:>8.3f}   {pred['phi']:>8.3f}   {(pred['phi']-phi_true)/phi_true*100:+7.1f}%
Проницаемость, мД   {K_true:>8.1f}   {pred['K']:>8.1f}   {(pred['K']-K_true)/K_true*100:+7.1f}%
Плотность, г/см³    {rho_true:>8.3f}   {pred['rho']:>8.3f}   {(pred['rho']-rho_true)/rho_true*100:+7.1f}%

Доверительный интервал прогноза (±σ деревьев RF):
  φ  : {pred['phi']-pred['phi_std']:.3f} … {pred['phi']+pred['phi_std']:.3f}
  K  : {pred['K_low']:.1f} … {pred['K_high']:.1f}  мД
  ρ  : {pred['rho']-pred['rho_std']:.3f} … {pred['rho']+pred['rho_std']:.3f}

Прогноз построен по медианам соседних скважин архива (Lab_A + Lab_B)
методом Random Forest по координатам (x, y)."""
                    self.compare_text.setPlainText(text)
                    return
            except Exception as e:
                self.compare_text.setPlainText(f"Не удалось построить прогноз: {e}")
                return
        if archive_core_clean is None:
            self.compare_text.setPlainText(
                "Для сравнения с прогнозом нужно сначала загрузить архив "
                "месторождения (кнопка «Загрузить архив»).")
        else:
            self.compare_text.setPlainText(
                "В файле скважины не указаны координаты x_m, y_m — "
                "прогноз по соседям невозможен.")


class RestoreTab(QWidget):
    """Вкладка 6: восстановление свойств по координатам / dropdown."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        title = QLabel("Восстановление свойств скважины по координатам")
        title.setObjectName('title')
        layout.addWidget(title)

        sub = QLabel("Random Forest, обученный на медианах соседних скважин")
        sub.setObjectName('subtitle')
        layout.addWidget(sub)

        # Два варианта ввода
        input_row = QHBoxLayout()
        input_row.setSpacing(20)

        # Вариант 1: из реестра
        gb1 = QGroupBox("Вариант 1: скважина без данных из реестра")
        gl1 = QVBoxLayout(gb1)
        self.combo = QComboBox()
        gl1.addWidget(self.combo)
        self.btn_combo = QPushButton("Восстановить")
        gl1.addWidget(self.btn_combo)
        gl1.addStretch()
        input_row.addWidget(gb1)

        # Вариант 2: вручную
        gb2 = QGroupBox("Вариант 2: ввести координаты вручную")
        gl2 = QVBoxLayout(gb2)
        row_xy = QHBoxLayout()
        row_xy.addWidget(QLabel("X, м:"))
        self.x_edit = QLineEdit("3000")
        row_xy.addWidget(self.x_edit)
        row_xy.addSpacing(10)
        row_xy.addWidget(QLabel("Y, м:"))
        self.y_edit = QLineEdit("2000")
        row_xy.addWidget(self.y_edit)
        gl2.addLayout(row_xy)
        self.btn_xy = QPushButton("Восстановить")
        gl2.addWidget(self.btn_xy)
        gl2.addStretch()
        input_row.addWidget(gb2)

        layout.addLayout(input_row)

        # Результат + карта
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QGroupBox("Результат прогноза")
        ll = QVBoxLayout(left)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        ll.addWidget(self.result_text)
        splitter.addWidget(left)

        right = QGroupBox("Карта пористости с прогнозом")
        rl = QVBoxLayout(right)
        self.canvas = MplCanvas(figsize=(7, 6))
        rl.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # Сигналы
        self.btn_combo.clicked.connect(self._restore_from_combo)
        self.btn_xy.clicked.connect(self._restore_from_xy)

        self._core_clean = None
        self._register = None
        self._cv = None

    def update_data(self, core_clean, register):
        self._core_clean = core_clean
        self._register = register

        # Дропдаун — скважины без файла из реестра
        self.combo.clear()
        if register is not None:
            no_data = register[~register['file_in_archive']]
            for _, row in no_data.iterrows():
                self.combo.addItem(
                    f"{row['well_id']}  (x={row['x_m']}, y={row['y_m']}, {row['zone']})",
                    userData=(row['well_id'], row['x_m'], row['y_m'], row['zone']))

        # LOO-CV для общей оценки качества
        if core_clean is not None and len(core_clean) > 0:
            self._cv = loowo_cv(core_clean)

    def _restore_from_combo(self):
        if self._core_clean is None or self.combo.count() == 0:
            self.result_text.setPlainText(
                "Сначала загрузите архив (кнопка «Загрузить архив»).")
            return
        data = self.combo.currentData()
        if data is None:
            return
        well_id, x, y, zone = data
        self._do_restore(well_id, x, y, zone)

    def _restore_from_xy(self):
        if self._core_clean is None:
            self.result_text.setPlainText(
                "Сначала загрузите архив (кнопка «Загрузить архив»).")
            return
        try:
            x = float(self.x_edit.text().replace(',', '.'))
            y = float(self.y_edit.text().replace(',', '.'))
        except ValueError:
            self.result_text.setPlainText("Введите корректные числа в X и Y.")
            return
        self._do_restore(f'X={x:.0f},Y={y:.0f}', x, y, '—')

    def _do_restore(self, label, x, y, zone):
        pred = predict_well_from_neighbors(self._core_clean, x, y)
        if pred is None:
            self.result_text.setPlainText("Недостаточно скважин для прогноза.")
            return

        cv_str = ""
        if self._cv:
            cv_str = (f"\nКачество модели (LOO-CV на {self._cv['n']} скважинах):\n"
                       f"  R² по пористости:    {self._cv['r2_phi']:+.3f}\n"
                       f"  R² по log K:         {self._cv['r2_K']:+.3f}\n"
                       f"  R² по плотности:     {self._cv['r2_rho']:+.3f}")

        self.result_text.setPlainText(f"""Целевая точка: {label}
Координаты: x = {x:.0f}, y = {y:.0f}
Зона: {zone}
Соседних скважин для обучения: {pred['n_neighbors']}

ПРОГНОЗ:
  Пористость, д.е.    {pred['phi']:.3f}   (±{pred['phi_std']:.3f})
  Проницаемость, мД   {pred['K']:.1f}    [{pred['K_low']:.1f} … {pred['K_high']:.1f}]
  Плотность, г/см³    {pred['rho']:.3f}   (±{pred['rho_std']:.3f})
{cv_str}

Доверительный интервал — стандартное отклонение прогнозов отдельных
деревьев Random Forest. Чем дальше точка от соседей, тем шире интервал.""")

        # Карта пористости с прогнозом
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)

        sub = self._core_clean[self._core_clean['lab_id'].isin(['Lab_A', 'Lab_B'])]
        ws = sub.groupby('well_id').agg(
            x=('x_m', 'first'), y=('y_m', 'first'),
            phi=('porosity', 'median'),
        ).reset_index()

        # Поверхность через RF
        X_tr = ws[['x', 'y']].values
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_tr, ws['phi'].values)

        xx, yy = np.meshgrid(
            np.linspace(ws['x'].min() - 200, ws['x'].max() + 200, 60),
            np.linspace(ws['y'].min() - 200, ws['y'].max() + 200, 60))
        zz = rf.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        cf = ax.contourf(xx, yy, zz, levels=20, cmap='RdYlGn', alpha=0.85)
        self.canvas.fig.colorbar(cf, ax=ax, fraction=0.04, label='φ')

        # Скважины с данными
        ax.scatter(ws['x'], ws['y'], c='#1a1a1a', s=35,
                   edgecolors='white', linewidth=1, zorder=3)
        # Скважины без данных
        if self._register is not None:
            no_data = self._register[~self._register['file_in_archive']]
            ax.scatter(no_data['x_m'], no_data['y_m'],
                       c='#cccccc', s=25, marker='s',
                       edgecolors='#777', linewidth=0.8, zorder=2)
        # Целевая
        ax.scatter([x], [y], c='#fbbf24', s=220, marker='*',
                   edgecolor='black', linewidth=1.5, zorder=10)
        ax.annotate(f'φ={pred["phi"]:.3f}',
                    (x, y), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='black', linewidth=0.8))

        ax.set_xlabel('X, м', fontsize=10)
        ax.set_ylabel('Y, м', fontsize=10)
        ax.set_title(f'Прогноз пористости: {label}', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


# =============================================================================
#                              ГЛАВНОЕ ОКНО
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Well Data Verifier — Березовское м/р")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        self._core = None         # core с флагами после verify
        self._core_clean = None    # только чистые строки
        self._register = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # ===== Шапка =====
        header = QHBoxLayout()
        title = QLabel("Well Data Verifier")
        title.setObjectName('title')
        sub = QLabel("Верификация и пространственное восстановление керновых данных")
        sub.setObjectName('subtitle')
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_block.addWidget(title)
        title_block.addWidget(sub)
        header.addLayout(title_block)
        header.addStretch()

        self.btn_archive = QPushButton("Загрузить архив")
        self.btn_well    = QPushButton("Загрузить скважину")
        self.btn_well.setObjectName('secondary')
        self.btn_export  = QPushButton("Экспорт отчёта")
        self.btn_export.setObjectName('secondary')
        self.btn_export.setEnabled(False)
        header.addWidget(self.btn_archive)
        header.addWidget(self.btn_well)
        header.addWidget(self.btn_export)

        root.addLayout(header)

        # Разделитель
        hr = QFrame()
        hr.setObjectName('hr')
        hr.setFixedHeight(1)
        root.addWidget(hr)

        # Статусная строка
        self.status = QLabel("Архив не загружен. Нажмите «Загрузить архив», чтобы начать.")
        self.status.setObjectName('subtitle')
        root.addWidget(self.status)

        # ===== Вкладки =====
        self.tabs = QTabWidget()
        self.tab_summary    = SummaryTab()
        self.tab_maps       = MapsTab()
        self.tab_profiles   = ProfilesTab()
        self.tab_variogram  = VariogramTab()
        self.tab_single     = SingleWellTab()
        self.tab_restore    = RestoreTab()

        self.tabs.addTab(self.tab_summary,   "Сводка")
        self.tabs.addTab(self.tab_maps,      "Карты")
        self.tabs.addTab(self.tab_profiles,  "Профили")
        self.tabs.addTab(self.tab_variogram, "Вариограмма")
        self.tabs.addTab(self.tab_single,    "Скважина")
        self.tabs.addTab(self.tab_restore,   "Восстановление")

        root.addWidget(self.tabs, 1)

        # Сигналы
        self.btn_archive.clicked.connect(self._on_load_archive)
        self.btn_well.clicked.connect(self._on_load_well)
        self.btn_export.clicked.connect(self._on_export)

    # -------- Слоты --------

    def _on_load_archive(self):
        d = QFileDialog.getExistingDirectory(self, "Папка архива (archive/)")
        if not d:
            return
        try:
            print(f"[1/6] Выбран путь: {d}")
            self.status.setText(f"Чтение файлов архива из {d}…")
            QApplication.processEvents()

            print(f"[2/6] Чтение CSV…")
            core_df, gis_df, register = load_archive(d)
            print(f"[3/6] Прочитано: core={core_df.shape}, gis={gis_df.shape}")

            self.status.setText("Верификация керна…")
            QApplication.processEvents()
            print(f"[4/6] Верификация…")
            core_with_flags, counts = verify_core(core_df)

            self._core = core_with_flags
            self._core_clean = core_with_flags[~core_with_flags['есть_ошибка']]
            self._register = register

            print(f"[5/6] Заполнение вкладок…")
            self.status.setText("Построение графиков…")
            QApplication.processEvents()

            self.tab_summary.update_data(core_with_flags, counts)
            QApplication.processEvents()
            self.tab_maps.update_data(core_with_flags, register)
            QApplication.processEvents()
            self.tab_profiles.update_data(core_with_flags)
            QApplication.processEvents()
            self.tab_variogram.update_data(core_with_flags)
            QApplication.processEvents()
            self.tab_restore.update_data(self._core_clean, register)
            QApplication.processEvents()

            self.btn_export.setEnabled(True)
            print(f"[6/6] Готово!")
            self.status.setText(
                f"Загружен архив: {core_df['well_id'].nunique()} скважин, "
                f"{len(core_df)} образцов керна, "
                f"{counts['аномалий']} аномалий.")
        except Exception as e:
            import traceback
            print(f"ОШИБКА: {e}")
            traceback.print_exc()
            self.status.setText(f"Ошибка загрузки архива: {e}")

    def _on_load_well(self):
        f, _ = QFileDialog.getOpenFileName(self, "Файл скважины",
                                            filter="CSV (*.csv);;All (*.*)")
        if not f:
            return
        try:
            well_df = load_single_well(f)
            self.tab_single.update_data(well_df, self._core_clean)
            self.tabs.setCurrentWidget(self.tab_single)
            self.status.setText(
                f"Загружена скважина: {os.path.basename(f)}  "
                f"({len(well_df)} образцов)")
        except Exception as e:
            self.status.setText(f"Ошибка загрузки скважины: {e}")

    def _on_export(self):
        if self._core is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт",
                                            "anomaly_report.csv",
                                            filter="CSV (*.csv)")
        if not f:
            return
        anom = self._core[self._core['есть_ошибка']]
        anom.to_csv(f, index=False, encoding='utf-8-sig')
        self.status.setText(f"Отчёт сохранён: {f} ({len(anom)} строк)")


# =============================================================================
#                                ЗАПУСК
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
