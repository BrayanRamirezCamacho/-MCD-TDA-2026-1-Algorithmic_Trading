
# ============================================================
# AUTO-INSTALL DEPENDENCIES IF MISSING
# ============================================================

import importlib
import subprocess
import sys

def install_if_missing(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing {package_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

required_packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("ripser", "ripser"),
    ("persim", "persim"),
    ("gudhi", "gudhi"),
    ("kagglehub", "kagglehub")
]

for pkg, imp in required_packages:
    install_if_missing(pkg, imp)

print("All dependencies ready.")


# ============================================================
# ORIGINAL NOTEBOOK CODE
# ============================================================


# ---------------- CELL 1 ----------------
pip install --upgrade kagglehub 

# ---------------- CELL 2 ----------------
# Instalar dependencias
! pip install -q ripser gudhi persim kagglehub[pandas-datasets]
print("✓ Dependencias instaladas")

# ---------------- CELL 3 ----------------
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Cargar el dataset completo (listado de archivos disponibles) ──────────────
import kagglehub, os

path = kagglehub.dataset_download("sudalairajkumar/cryptocurrencypricehistory")
print("Archivos disponibles:")
archivos = sorted(os.listdir(path))
for f in archivos:
    print(f"  {f}")

# ---------------- CELL 4 ----------------
# ── Seleccionar monedas para el análisis ──────────────────────────────────────
# Ajusta esta lista según los archivos disponibles
COINS = {
    'BTC': 'coin_Bitcoin.csv',
    'ETH': 'coin_Ethereum.csv',
    'BNB': 'coin_BinanceCoin.csv',
    'ADA': 'coin_Cardano.csv',
    'XRP': 'coin_XRP.csv',
}

dfs = {}
for ticker, fname in COINS.items():
    fpath = os.path.join(path, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        dfs[ticker] = df
        print(f"  {ticker}: {len(df)} filas  ({df['Date'].min().date()} → {df['Date'].max().date()})")
    else:
        print(f"  {ticker}: archivo no encontrado ({fname})")

# Usar BTC como referencia principal
df_btc = dfs['BTC'].copy()
print(f"\nColumnas: {df_btc.columns.tolist()}")
df_btc.head(3)

# ---------------- CELL 5 ----------------
# ── CONFIG ─────────────────────────────────────────────────────────────────────
WINDOW     = 30      # tamaño de la ventana deslizante (días)
STEP_WIN   = 5       # paso entre ventanas
TAU        = 1       # retardo de Takens
DIM_TAKENS = 3       # dimensión del encaje
EPS_MAX    = 3.0     # máximo ε para CROCKER
N_EPS      = 60      # puntos en la grilla de ε
MAX_FRAMES = 80      # máximo frames para CROCKER

# ── Features por ventana ───────────────────────────────────────────────────────
def build_features(df):
    """Construye las columnas de features a partir de OHLCV."""
    d = df.copy()
    d['log_ret']  = np.log(d['Close'] / d['Close'].shift(1))
    d['log_vol']  = np.log(d['Volume'] + 1)
    d['hl_range'] = (d['High'] - d['Low']) / d['Close']  # rango normalizado
    d['log_price']= np.log(d['Close'])
    d = d.dropna().reset_index(drop=True)
    return d

df_btc = build_features(df_btc)

# ── Encaje de Takens sobre log-retornos ───────────────────────────────────────
def takens_embedding(series, tau=1, dim=3):
    """
    Encaje con retardo: convierte una serie 1D en una nube de puntos (N, dim).
    Cada punto es [x(t), x(t+tau), x(t+2*tau), ...]
    """
    n = len(series) - (dim - 1) * tau
    if n <= 0:
        return np.zeros((1, dim))
    pts = np.stack([series[i * tau: i * tau + n] for i in range(dim)], axis=1)
    return pts

# ── Nube de puntos multivariada (precio + volumen + rango) ───────────────────
def build_cloud_multivariate(df_window):
    """
    Construye una nube de puntos (N, 4) combinando:
    log_ret (encaje Takens dim=2) + log_vol + hl_range
    """
    ret  = df_window['log_ret'].values
    vol  = df_window['log_vol'].values
    hlr  = df_window['hl_range'].values

    n = len(ret)
    if n < 3:
        return np.zeros((1, 4))

    # Takens dim=2 sobre log_ret
    tk = takens_embedding(ret, tau=TAU, dim=2)  # (n-tau, 2)
    m  = len(tk)

    cloud = np.column_stack([
        tk,
        vol[TAU:TAU + m],
        hlr[TAU:TAU + m],
    ])

    # Normalización z-score por columna
    mu  = cloud.mean(axis=0)
    std = cloud.std(axis=0) + 1e-8
    cloud = (cloud - mu) / std
    return cloud

# ── Crear ventanas ────────────────────────────────────────────────────────────
windows, window_dates = [], []
for start in range(0, len(df_btc) - WINDOW, STEP_WIN):
    end  = start + WINDOW
    chunk = df_btc.iloc[start:end]
    cloud = build_cloud_multivariate(chunk)
    windows.append(cloud)
    window_dates.append(df_btc.iloc[end - 1]['Date'])

print(f"Ventanas creadas  : {len(windows)}")
print(f"Rango de fechas   : {window_dates[0].date()} → {window_dates[-1].date()}")
print(f"Nube de ejemplo   : {windows[0].shape}")

# ---------------- CELL 6 ----------------
from ripser import ripser
import time

def compute_ph(cloud, maxdim=1):
    """
    Calcula homología persistente de Vietoris-Rips sobre una nube de puntos.
    Devuelve lista de diagramas: dgms[0]=H0, dgms[1]=H1
    """
    result = ripser(cloud, maxdim=maxdim)
    return result['dgms']

print("Calculando PH en todas las ventanas...")
t0 = time.time()

all_dgms = []
for i, cloud in enumerate(windows):
    dgms = compute_ph(cloud, maxdim=1)
    all_dgms.append(dgms)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(windows)} ventanas procesadas")

print(f"\n✓ PH completado en {time.time() - t0:.1f}s")
print(f"  H0 bars (ventana 0): {len(all_dgms[0][0])}")
print(f"  H1 bars (ventana 0): {len(all_dgms[0][1])}")

# ---------------- CELL 7 ----------------
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size': 11,
})

def plot_barcodes(dgms, eps_max=EPS_MAX, title='Barcodes', date_label=''):
    """
    Barcode clásico: H0 arriba (azul), H1 abajo (rojo).
    Barras más largas = características más persistentes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    palette   = ['#1f77b4', '#d62728']
    labels    = ['H₀  (componentes conexas)', 'H₁  (ciclos topológicos)']

    for k, (ax, col, lab) in enumerate(zip(axes, palette, labels)):
        dgm  = dgms[k].copy()
        # Reemplazar infinito por eps_max
        dgm[np.isinf(dgm[:, 1]), 1] = eps_max
        # Ordenar por persistencia descendente
        order = np.argsort(dgm[:, 1] - dgm[:, 0])[::-1]
        dgm   = dgm[order]

        for i, (birth, death) in enumerate(dgm):
            ax.plot([birth, death], [i, i],
                    color=col, lw=2.0, solid_capstyle='butt')

        ax.set_ylabel(lab, fontsize=10.5)
        ax.set_xlim(-0.02, eps_max)
        ax.set_ylim(-0.5, max(len(dgm) - 0.5, 0.5))
        ax.set_yticks([])
        ax.axvline(0, color='gray', lw=0.5, ls='--')

    axes[-1].set_xlabel(r'Parámetro de proximidad $\varepsilon$', fontsize=11)
    sub = f' — {date_label}' if date_label else ''
    fig.suptitle(title + sub, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig

# Graficar primera ventana
fig_bc = plot_barcodes(
    all_dgms[0],
    title='Barcodes BTC/USDT',
    date_label=str(window_dates[0].date())
)
plt.savefig('barcode_btc_t0.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: barcode_btc_t0.png")

# ---------------- CELL 8 ----------------
# ── Panel de barcodes: t=0, t=N/2, t=-1 ────────────────────────────────────
idxs   = [0, len(all_dgms)//2, -1]
titles = ['Inicio', 'Mitad', 'Fin']

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')
palette   = ['#1f77b4', '#d62728']
h_labels  = ['H₀', 'H₁']

for row, (idx, ttl) in enumerate(zip(idxs, titles)):
    dgms = all_dgms[idx]
    dstr = str(window_dates[idx].date())
    for col in range(2):
        ax  = axes[row, col]
        dgm = dgms[col].copy()
        dgm[np.isinf(dgm[:, 1]), 1] = EPS_MAX
        order = np.argsort(dgm[:, 1] - dgm[:, 0])[::-1]
        dgm   = dgm[order]
        for i, (b, d) in enumerate(dgm):
            ax.plot([b, d], [i, i], color=palette[col], lw=1.8, solid_capstyle='butt')
        ax.set_yticks([])
        ax.set_xlim(0, EPS_MAX)
        ax.set_ylim(-0.5, max(len(dgm) - 0.5, 0.5))
        ax.spines[['top','right']].set_visible(False)
        if row == 0:
            ax.set_title(h_labels[col], fontsize=12, color=palette[col], fontweight='bold')
        ax.set_ylabel(f'{ttl}\n{dstr}', fontsize=9)

fig.suptitle('Barcodes BTC/USDT — evolución temporal', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig('barcodes_panel.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 9 ----------------
def plot_persistence_diagram(dgms, eps_max=EPS_MAX, title='Diagrama de persistencia', date_label=''):
    """
    Birth vs death scatter.
    - Puntos sobre la diagonal = ruido topológico
    - Puntos lejos de la diagonal = características persistentes (señal)
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    palette = {0: '#1f77b4', 1: '#d62728'}
    markers = {0: 'o',       1: 's'}
    mlabels = {0: 'H₀ (componentes)', 1: 'H₁ (ciclos)'}

    for k in range(min(2, len(dgms))):
        dgm = dgms[k].copy()
        dgm[np.isinf(dgm[:, 1]), 1] = eps_max
        # Tamaño proporcional a la persistencia
        pers = dgm[:, 1] - dgm[:, 0]
        sizes = 20 + 80 * (pers / (pers.max() + 1e-9))
        ax.scatter(dgm[:, 0], dgm[:, 1],
                   c=palette[k], marker=markers[k],
                   s=sizes, alpha=0.80,
                   label=mlabels[k], zorder=3, edgecolors='none')

    lim = eps_max * 1.07
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='diagonal', zorder=1)
    # Zona de ruido (bajo la diagonal)
    ax.fill_between([0, lim], [0, lim], 0,
                    color='gray', alpha=0.04, zorder=0)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Nacimiento $b$', fontsize=11)
    ax.set_ylabel(r'Muerte $d$', fontsize=11)
    sub = f' — {date_label}' if date_label else ''
    ax.set_title(title + sub, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig

fig_pd = plot_persistence_diagram(
    all_dgms[0],
    title='Diagrama de persistencia BTC/USDT',
    date_label=str(window_dates[0].date())
)
plt.savefig('persistence_diagram_t0.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 10 ----------------
# ── Panel 2x2: 4 momentos del tiempo ──────────────────────────────────────────
n_panels = min(4, len(all_dgms))
idxs_pd  = np.linspace(0, len(all_dgms) - 1, n_panels, dtype=int)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, idx in zip(axes, idxs_pd):
    dgms = all_dgms[idx]
    dstr = str(window_dates[idx].date())
    palette = {0: '#1f77b4', 1: '#d62728'}
    for k in range(2):
        dgm = dgms[k].copy()
        dgm[np.isinf(dgm[:, 1]), 1] = EPS_MAX
        pers  = dgm[:, 1] - dgm[:, 0]
        sizes = 15 + 60 * (pers / (pers.max() + 1e-9))
        ax.scatter(dgm[:, 0], dgm[:, 1], c=palette[k],
                   s=sizes, alpha=0.75, zorder=3,
                   label='H₀' if k == 0 else 'H₁')
    lim = EPS_MAX * 1.07
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_title(dstr, fontsize=10)
    ax.set_xlabel('Nacimiento', fontsize=9)
    ax.set_ylabel('Muerte', fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

fig.suptitle('Diagramas de persistencia BTC/USDT — evolución', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig('persistence_panel.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 11 ----------------
from matplotlib.lines import Line2D

eps_vals = np.linspace(0.0, EPS_MAX, N_EPS)

def betti_vs_epsilon(dgms, eps_vals):
    """Evalúa b0(ε) y b1(ε) en una grilla discreta."""
    cap = eps_vals[-1] + 1.0
    b0  = np.zeros(len(eps_vals), dtype=int)
    b1  = np.zeros(len(eps_vals), dtype=int)
    for bars, bk in zip(dgms[:2], [b0, b1]):
        for birth, death in bars:
            if np.isinf(death):
                death = cap
            mask = (eps_vals >= birth) & (eps_vals < death)
            bk[mask] += 1
    return b0, b1

# ── Construir matrices CROCKER ─────────────────────────────────────────────────
step_c = max(1, len(all_dgms) // MAX_FRAMES)
t_idxs = list(range(0, len(all_dgms), step_c))

B0 = np.zeros((N_EPS, len(t_idxs)))
B1 = np.zeros((N_EPS, len(t_idxs)))

for col, ti in enumerate(t_idxs):
    b0, b1 = betti_vs_epsilon(all_dgms[ti], eps_vals)
    B0[:, col] = b0
    B1[:, col] = b1

times_used = [window_dates[ti] for ti in t_idxs]
print(f"Matrices CROCKER: {B0.shape}  (eps × tiempo)")

# ---------------- CELL 12 ----------------
def plot_crocker(B, eps_vals, times, title='CROCKER', max_level=5):
    """
    CROCKER contour plot: cada contorno delimita la región donde
    b_k(ε, t) = k. Replica Topaz et al. (2015) Fig 6.
    """
    level_colors = {
        1: '#d62728',  # rojo
        2: '#bcbd22',  # amarillo-verde
        3: '#2ca02c',  # verde
        4: '#1f77b4',  # azul
        5: '#9467bd',  # púrpura
    }
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(14, 4.5))
    T_num = mdates.date2num(times)

    # Región de ruido (b ≥ max_level+1)
    noise = (B >= max_level + 1).astype(float)
    if noise.max() > 0:
        ax.contourf(T_num, eps_vals, noise,
                    levels=[0.5, 1.5],
                    colors=['#c4b5fd'], alpha=0.30, zorder=1)

    handles = []
    for level in range(1, max_level + 1):
        col = level_colors.get(level, 'black')
        try:
            ax.contour(T_num, eps_vals, B,
                       levels=[level - 0.5],
                       colors=[col], linewidths=1.8, zorder=2 + level)
        except Exception:
            pass
        handles.append(Line2D([0],[0], color=col, lw=2.2, label=str(level)))

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate(rotation=30)

    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel(r'Parámetro $\varepsilon$', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(handles=handles, title='nivel',
              loc='upper right', fontsize=9, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig

fig_c0 = plot_crocker(B0, eps_vals, times_used,
                      title=r'CROCKER — $b_0(\varepsilon, t)$ — BTC/USDT',
                      max_level=5)
plt.savefig('crocker_b0.png', dpi=150, bbox_inches='tight')
plt.show()

fig_c1 = plot_crocker(B1, eps_vals, times_used,
                      title=r'CROCKER — $b_1(\varepsilon, t)$ — BTC/USDT',
                      max_level=4)
plt.savefig('crocker_b1.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 13 ----------------
import gudhi.wasserstein

def wasserstein_dist(d1, d2, order=1, internal_p=2):
    """Distancia de Wasserstein entre dos diagramas de persistencia."""
    if len(d1) == 0 or len(d2) == 0:
        return 0.0
    return gudhi.wasserstein.wasserstein_distance(
        d1, d2, order=order, internal_p=internal_p
    )

# ── Calcular distancias consecutivas (dinámica temporal) ─────────────────────
print("Calculando distancias Wasserstein consecutivas...")
wass_results = {}
for order in [1, 2, np.inf]:
    for dim in [0, 1]:
        key = f'p{order}_b{dim}'
        dists = []
        for i in range(1, len(all_dgms)):
            d1 = all_dgms[i-1][dim]
            d2 = all_dgms[i][dim]
            dists.append(wasserstein_dist(d1, d2, order=order))
        wass_results[key] = dists

print(f"✓ Listo. {len(wass_results)} series de distancias calculadas.")

# ---------------- CELL 14 ----------------
import matplotlib.dates as mdates

dates_wass = window_dates[1:]  # un paso menos que ventanas

def plot_wasserstein_dynamics(wass_results, dates, coins_label='BTC/USDT'):
    """
    Paneles de dinámica Wasserstein para órdenes p=1, p=2, p=inf
    y dimensiones H0, H1.
    Picos = cambios abruptos de régimen.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    orders = [1, 2, np.inf]
    colors_h = ['#1f77b4', '#d62728']

    for row, order in enumerate(orders):
        for col, dim in enumerate([0, 1]):
            ax  = axes[row, col]
            key = f'p{order}_b{dim}'
            d   = np.array(wass_results[key])

            # Suavizado para detectar régimen
            smooth = pd.Series(d).rolling(5, center=True, min_periods=1).mean().values

            ax.fill_between(dates, d, alpha=0.25, color=colors_h[col])
            ax.plot(dates, d, lw=0.8, alpha=0.6, color=colors_h[col])
            ax.plot(dates, smooth, lw=2.0, color=colors_h[col], label='suavizado')

            # Marcar picos (posibles cambios de régimen)
            threshold = np.mean(d) + 1.5 * np.std(d)
            peaks_idx = np.where(d > threshold)[0]
            if len(peaks_idx):
                ax.scatter([dates[i] for i in peaks_idx], d[peaks_idx],
                           color='black', s=25, zorder=5,
                           label=f'pico (>{threshold:.3f})')
            ax.axhline(threshold, color='gray', ls=':', lw=1)

            ord_lbl = '∞' if order == np.inf else str(order)
            ax.set_title(f'W$_{{p={ord_lbl}}}$ — H{dim}', fontsize=11)
            ax.set_ylabel('Distancia', fontsize=9)
            ax.legend(fontsize=8)
            ax.spines[['top','right']].set_visible(False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig.autofmt_xdate(rotation=30)
    fig.suptitle(f'Dinámica Wasserstein — {coins_label}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig

fig_wd = plot_wasserstein_dynamics(wass_results, dates_wass)
plt.savefig('wasserstein_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 15 ----------------
# ── Calcular matriz pairwise (submuestra para eficiencia) ─────────────────────
HMAP_FRAMES = 40  # reducir si es lento
step_h  = max(1, len(all_dgms) // HMAP_FRAMES)
h_idxs  = list(range(0, len(all_dgms), step_h))[:HMAP_FRAMES]
n_h     = len(h_idxs)
h_dates = [window_dates[i] for i in h_idxs]

def distance_heatmap(all_dgms, h_idxs, order=1, dim=1):
    """Matriz de distancias Wasserstein pairwise."""
    n = len(h_idxs)
    M = np.zeros((n, n))
    for i, ti in enumerate(h_idxs):
        for j, tj in enumerate(h_idxs):
            if i <= j:
                v = wasserstein_dist(all_dgms[ti][dim], all_dgms[tj][dim], order=order)
                M[i, j] = v
                M[j, i] = v
    return M

print("Calculando heatmap W1-H1 (esto puede tardar un momento)...")
M_w1_h1 = distance_heatmap(all_dgms, h_idxs, order=1, dim=1)
print(f"✓ Matriz: {M_w1_h1.shape}")

# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_heatmap(M, dates, title='Heatmap Wasserstein'):
    tick_labels = [d.strftime('%Y-%m') for d in dates]
    step_tck    = max(1, len(dates) // 8)
    tck_idxs    = list(range(0, len(dates), step_tck))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(M, cmap='inferno', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='W₁(H₁)')
    ax.set_xticks(tck_idxs)
    ax.set_xticklabels([tick_labels[i] for i in tck_idxs], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(tck_idxs)
    ax.set_yticklabels([tick_labels[i] for i in tck_idxs], fontsize=8)
    ax.set_xlabel('Ventana temporal', fontsize=11)
    ax.set_ylabel('Ventana temporal', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig

fig_hm = plot_heatmap(M_w1_h1, h_dates,
                      title='Heatmap Wasserstein — BTC/USDT (W₁, H₁)')
plt.savefig('heatmap_wasserstein.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 16 ----------------
def persistent_entropy(bars):
    """
    Entropía de Shannon sobre la distribución de persistencias.
    Ref: Piangerelli et al. (2018).
    Valor alto = heterogeneidad topológica (mercado caótico)
    Valor bajo = organización emergente (tendencia fuerte)
    """
    finite = bars[~np.isinf(bars[:, 1])]
    if len(finite) == 0:
        return 0.0
    lens  = finite[:, 1] - finite[:, 0]
    total = lens.sum()
    if total == 0:
        return 0.0
    p = lens / total
    return -np.sum(p * np.log(p + 1e-12))

ent_h0 = np.array([persistent_entropy(d[0]) for d in all_dgms])
ent_h1 = np.array([persistent_entropy(d[1]) for d in all_dgms])

# Correlacionar con log-retorno de BTC
log_ret_wins = [df_btc.iloc[i * STEP_WIN: i * STEP_WIN + WINDOW]['log_ret'].sum()
                for i in range(len(all_dgms))]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel 1: precio BTC
close_wins = [df_btc.iloc[i * STEP_WIN + WINDOW - 1]['Close'] for i in range(len(all_dgms))]
axes[0].plot(window_dates, close_wins, color='#1f77b4', lw=1.5)
axes[0].set_ylabel('Precio BTC (USD)', fontsize=10)
axes[0].set_title('Entropía persistente vs precio BTC', fontsize=13, fontweight='bold')
axes[0].fill_between(window_dates, close_wins, alpha=0.15, color='#1f77b4')
axes[0].spines[['top','right']].set_visible(False)

# Panel 2: entropía H0 y H1
axes[1].plot(window_dates, ent_h0, color='#1f77b4', lw=1.5, label='H₀')
axes[1].plot(window_dates, ent_h1, color='#d62728', lw=1.5, label='H₁')
axes[1].set_ylabel('Entropía persistente', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].spines[['top','right']].set_visible(False)

# Panel 3: log-retorno acumulado por ventana
axes[2].bar(window_dates, log_ret_wins,
            color=['#2ca02c' if r > 0 else '#d62728' for r in log_ret_wins],
            width=STEP_WIN * 0.8, alpha=0.7)
axes[2].set_ylabel('Log-retorno acum. ventana', fontsize=10)
axes[2].axhline(0, color='black', lw=0.7)
axes[2].spines[['top','right']].set_visible(False)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.autofmt_xdate(rotation=30)
fig.tight_layout()
plt.savefig('entropy_vs_price.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Correlación entropía H1 vs log-ret: {np.corrcoef(ent_h1, log_ret_wins)[0,1]:.3f}")

# ---------------- CELL 17 ----------------
from persim import PersistenceImager

pimgr = PersistenceImager(
    birth_range=(0, EPS_MAX),
    pers_range=(0, EPS_MAX),
    pixel_size=0.15,
    weight='persistence',   # ← cambiar 'linear' por 'persistence'
)

# Calcular con H1 (más informativa para ciclos de mercado)
h1_bars = [d[1] for d in all_dgms]

# Filtrar diagramas con barras finitas
h1_finite = []
for bars in h1_bars:
    finite = bars[~np.isinf(bars[:, 1])]
    if len(finite) == 0:
        finite = np.array([[0.0, 0.01]])
    h1_finite.append(finite)

pimgr.fit(h1_finite)
imgs = np.array(pimgr.transform(h1_finite))  # ← envolver con np.array()
print(f"Imágenes de persistencia: {imgs.shape}")

# ── Visualizar 6 imágenes ──────────────────────────────────────────────────────
n_show  = 6
idxs_pi = np.linspace(0, len(imgs) - 1, n_show, dtype=int)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for ax, idx in zip(axes, idxs_pi):
    im = ax.imshow(imgs[idx], origin='lower', cmap='hot',
                   extent=[0, EPS_MAX, 0, EPS_MAX], aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(window_dates[idx].strftime('%Y-%m-%d'), fontsize=10)
    ax.set_xlabel('Nacimiento', fontsize=8)
    ax.set_ylabel('Persistencia', fontsize=8)

fig.suptitle('Imágenes de persistencia H₁ — BTC/USDT', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig('persistence_images.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 18 ----------------
# ── Construir DataFrame de features topológicas ────────────────────────────────
def total_persistence(bars, p=1):
    """Persistencia total L^p."""
    finite = bars[~np.isinf(bars[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float(np.sum((finite[:, 1] - finite[:, 0]) ** p) ** (1.0 / p))

def betti_at_eps(dgm, eps):
    """Número de Betti b(ε) para un ε fijo."""
    count = 0
    for birth, death in dgm:
        d = death if not np.isinf(death) else EPS_MAX + 1
        if birth <= eps < d:
            count += 1
    return count

EPS_REF = 0.5  # ε de referencia para Betti puntual

records = []
for i, dgms in enumerate(all_dgms):
    h0, h1 = dgms
    # Wasserstein consecutivo (ya calculado)
    w1_h0 = wass_results['p1_b0'][i - 1] if i > 0 else 0.0
    w1_h1 = wass_results['p1_b1'][i - 1] if i > 0 else 0.0
    w2_h1 = wass_results['p2_b1'][i - 1] if i > 0 else 0.0

    records.append({
        'date'          : window_dates[i],
        # Entropía persistente
        'ent_h0'        : persistent_entropy(h0),
        'ent_h1'        : persistent_entropy(h1),
        # Persistencia total
        'total_pers_h0' : total_persistence(h0),
        'total_pers_h1' : total_persistence(h1),
        # Número de barras
        'n_bars_h0'     : len(h0),
        'n_bars_h1'     : len(h1),
        # Número de Betti a ε de referencia
        'betti0_ref'    : betti_at_eps(h0, EPS_REF),
        'betti1_ref'    : betti_at_eps(h1, EPS_REF),
        # Máxima persistencia
        'max_pers_h0'   : float(np.max(h0[~np.isinf(h0[:,1]), 1] - h0[~np.isinf(h0[:,1]), 0])) if len(h0) > 1 else 0,
        'max_pers_h1'   : float(np.max(h1[~np.isinf(h1[:,1]), 1] - h1[~np.isinf(h1[:,1]), 0])) if len(h1) > 0 and not np.all(np.isinf(h1[:,1])) else 0,
        # Distancias Wasserstein
        'wass1_h0'      : w1_h0,
        'wass1_h1'      : w1_h1,
        'wass2_h1'      : w2_h1,
    })

df_tda = pd.DataFrame(records).set_index('date')

roll = 25

# Wasserstein z-score (alerta de régimen)
df_tda['wass1_h1_zscore'] = (
    (df_tda['wass1_h1'] - df_tda['wass1_h1'].rolling(roll).mean()) /
    (df_tda['wass1_h1'].rolling(roll).std() + 1e-9)
)
df_tda['regime_alert'] = (df_tda['wass1_h1_zscore'].abs() > 2.0).astype(int)

# Suavizar entropía antes del z-score
df_tda['ent_h1_smooth'] = df_tda['ent_h1'].rolling(5, center=True).mean()

# Z-score sobre entropía suavizada
df_tda['ent_h1_zscore'] = (
    (df_tda['ent_h1_smooth'] - df_tda['ent_h1_smooth'].rolling(roll).mean()) /
    (df_tda['ent_h1_smooth'].rolling(roll).std() + 1e-9)
)
df_tda['tda_long_signal']  = (df_tda['ent_h1_zscore'] < -2.0).astype(int)
df_tda['tda_short_signal'] = (df_tda['ent_h1_zscore'] >  2.0).astype(int)

df_tda.dropna(inplace=True)

# ---------------- CELL 19 ----------------
# ── Visualización de señales TDA vs precio ─────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# Precio
close_idx = df_btc.set_index('Date')['Close'].reindex(df_tda.index, method='nearest')
axes[0].semilogy(df_tda.index, close_idx, color='#1f77b4', lw=1.5)
# Marcar señales
long_dates  = df_tda[df_tda['tda_long_signal']  == 1].index
short_dates = df_tda[df_tda['tda_short_signal'] == 1].index
axes[0].scatter(long_dates,  close_idx.reindex(long_dates),  color='#2ca02c', s=40, zorder=5, label='señal LONG')
axes[0].scatter(short_dates, close_idx.reindex(short_dates), color='#d62728', s=40, zorder=5, marker='v', label='señal SHORT')
axes[0].set_ylabel('BTC/USDT (log)', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].set_title('Señales TDA sobre precio BTC/USDT', fontsize=13, fontweight='bold')
axes[0].spines[['top','right']].set_visible(False)

# Wasserstein z-score
axes[1].plot(df_tda.index, df_tda['wass1_h1_zscore'], color='#9467bd', lw=1.5)
axes[1].axhline(1.5,  color='#d62728', ls='--', lw=1, label='+1.5σ')
axes[1].axhline(-1.5, color='#2ca02c', ls='--', lw=1, label='-1.5σ')
axes[1].fill_between(df_tda.index, df_tda['wass1_h1_zscore'], 0,
                     where=df_tda['regime_alert'] == 1, alpha=0.3, color='orange', label='alerta')
axes[1].set_ylabel('W₁(H₁) z-score', fontsize=10)
axes[1].legend(fontsize=8)
axes[1].spines[['top','right']].set_visible(False)

# Entropía H1
axes[2].plot(df_tda.index, df_tda['ent_h1'], color='#d62728', lw=1.5)
axes[2].fill_between(df_tda.index, df_tda['ent_h1'],
                     alpha=0.2, color='#d62728')
axes[2].set_ylabel('Entropía H₁', fontsize=10)
axes[2].spines[['top','right']].set_visible(False)

# Número de barras H1
axes[3].bar(df_tda.index, df_tda['n_bars_h1'],
            width=STEP_WIN * 0.8, color='#17becf', alpha=0.7)
axes[3].set_ylabel('Barras H₁', fontsize=10)
axes[3].spines[['top','right']].set_visible(False)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.autofmt_xdate(rotation=30)
fig.tight_layout()
plt.savefig('tda_signals.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 20 ----------------
# ── Construir features TDA para todas las monedas ─────────────────────────────
def tda_features_for_coin(df_coin, window=WINDOW, step=STEP_WIN):
    """Pipeline TDA completo para una moneda."""
    df = build_features(df_coin)
    if len(df) < window + 5:
        return None, []
    wins, wdates = [], []
    for start in range(0, len(df) - window, step):
        end   = start + window
        cloud = build_cloud_multivariate(df.iloc[start:end])
        wins.append(cloud)
        wdates.append(df.iloc[end - 1]['Date'])
    dgms_all = [compute_ph(c, maxdim=1) for c in wins]
    records  = []
    for i, dgms in enumerate(dgms_all):
        records.append({
            'date'     : wdates[i],
            'ent_h1'   : persistent_entropy(dgms[1]),
            'tot_h1'   : total_persistence(dgms[1]),
            'n_h1'     : len(dgms[1]),
        })
    return pd.DataFrame(records).set_index('date'), dgms_all

coin_features = {}
coin_dgms     = {}
for ticker, df_coin in dfs.items():
    print(f"  {ticker}...", end=' ')
    feat, dgms = tda_features_for_coin(df_coin)
    if feat is not None:
        coin_features[ticker] = feat
        coin_dgms[ticker]     = dgms
        print(f"{len(feat)} ventanas")
    else:
        print("insuficiente data")

# ── Comparar entropía H1 entre monedas ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
colors_coins = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd']
for (ticker, feat), col in zip(coin_features.items(), colors_coins):
    ax.plot(feat.index, feat['ent_h1'], label=ticker, lw=1.5, color=col)

ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Entropía persistente H₁', fontsize=11)
ax.set_title('Entropía persistente H₁ — comparación multi-moneda', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.spines[['top','right']].set_visible(False)
fig.autofmt_xdate(rotation=30)
fig.tight_layout()
plt.savefig('multicoins_entropy.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- CELL 21 ----------------
# ── Exportar CSV de features TDA ──────────────────────────────────────────────
df_tda.to_csv('tda_features_btc.csv')
print("✓ Guardado: tda_features_btc.csv")
print(f"  Columnas: {df_tda.columns.tolist()}")
print(f"  Filas   : {len(df_tda)}")

# ── Resumen estadístico ────────────────────────────────────────────────────────
print("\n=== Resumen de features TDA ===")
print(df_tda[['ent_h0','ent_h1','wass1_h1','total_pers_h1',
              'regime_alert','tda_long_signal']].describe().round(4))

print(f"\n Señales LONG  : {df_tda['tda_long_signal'].sum()} de {len(df_tda)} ventanas")
print(f" Señales SHORT : {df_tda['tda_short_signal'].sum()} de {len(df_tda)} ventanas")
print(f" Alertas régimen: {df_tda['regime_alert'].sum()} de {len(df_tda)} ventanas")

print("\n=== Archivos generados ===")
import glob
for f in sorted(glob.glob('*.png') + glob.glob('*.csv')):
    print(f"  {f}")
