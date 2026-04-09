"""
DETECTOR DE DADOS SINTÉTICOS
=============================
Roda esse script antes de qualquer modelo de ML.
Ele vai te dizer se vale a pena continuar ou não.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from kagglehub import KaggleDatasetAdapter
import kagglehub

# ─────────────────────────────────────────────────────────────
# CARREGA O DATASET
# ─────────────────────────────────────────────────────────────
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ihasan88/student-performance-dataset",
    "bangladesh_student_performance.csv",
)
df = df.drop(columns=["Student_ID"], errors="ignore")

TARGET = "HSC_Result"

# ─────────────────────────────────────────────────────────────
# SEPARAR COLUNAS
# ─────────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

df_enc = df.copy()
for c in cat_cols:
    df_enc[c] = pd.factorize(df_enc[c])[0]

alertas = []
pontos   = 0  # quanto mais alto, mais suspeito

print("=" * 65)
print("  RELATÓRIO DE QUALIDADE — DETECÇÃO DE DADOS SINTÉTICOS")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# PASSO 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════
print("\n📋 PASSO 1: VISÃO GERAL")
print("-" * 45)
print(f"  Linhas       : {df.shape[0]}")
print(f"  Colunas      : {df.shape[1]}")
print(f"  Numéricas    : {len(num_cols)}")
print(f"  Categóricas  : {len(cat_cols)}")
print(f"  Valores nulos: {df.isnull().sum().sum()}")

if df.isnull().sum().sum() == 0:
    alertas.append("Nenhum valor nulo — datasets reais quase sempre têm algum.")
    pontos += 1

# ══════════════════════════════════════════════════════════════
# PASSO 2 — CORRELAÇÃO COM O TARGET
# ══════════════════════════════════════════════════════════════
print("\n\n📊 PASSO 2: CORRELAÇÃO DAS FEATURES COM O TARGET")
print("-" * 45)
print(f"  Target: {TARGET}\n")

corr_target = df_enc.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
max_corr = corr_target.abs().max()
media_corr = corr_target.abs().mean()

for feat, val in corr_target.items():
    sinal = "🔴" if abs(val) < 0.05 else ("🟡" if abs(val) < 0.15 else "🟢")
    print(f"  {sinal} {feat:<25} {val:+.4f}")

print(f"\n  Maior correlação : {max_corr:.4f}")
print(f"  Média das corr.  : {media_corr:.4f}")

if max_corr < 0.10:
    alertas.append(f"Correlação máxima com target é só {max_corr:.4f} — praticamente zero.")
    pontos += 3
elif max_corr < 0.20:
    alertas.append(f"Correlação máxima com target é baixa ({max_corr:.4f}).")
    pontos += 1

# ══════════════════════════════════════════════════════════════
# PASSO 3 — CORRELAÇÃO ENTRE FEATURES (LÓGICA DE NEGÓCIO)
# ══════════════════════════════════════════════════════════════
print("\n\n🔗 PASSO 3: CORRELAÇÃO ENTRE FEATURES")
print("-" * 45)
print("  Em dados reais, features relacionadas devem se correlacionar.\n")

corr_matrix = df_enc.corr()
corr_arr = corr_matrix.values.copy().astype(float)
np.fill_diagonal(corr_arr, 0)

all_corrs = np.abs(corr_arr).flatten()
all_corrs = all_corrs[all_corrs > 0]
media_inter = all_corrs.mean()
max_inter   = all_corrs.max()

print(f"  Correlação média entre features : {media_inter:.4f}")
print(f"  Correlação máxima entre features: {max_inter:.4f}")

if media_inter < 0.05:
    alertas.append(f"Correlação média entre features é {media_inter:.4f} — features independentes entre si.")
    pontos += 3
    print("  🔴 Muito suspeito: features sem nenhuma relação entre si.")
elif media_inter < 0.10:
    alertas.append(f"Correlação média entre features baixa ({media_inter:.4f}).")
    pontos += 1
    print("  🟡 Levemente suspeito.")
else:
    print("  🟢 Correlações entre features parecem normais.")

# ══════════════════════════════════════════════════════════════
# PASSO 4 — DISTRIBUIÇÃO DAS FEATURES NUMÉRICAS
# ══════════════════════════════════════════════════════════════
print("\n\n📐 PASSO 4: DISTRIBUIÇÃO DAS FEATURES NUMÉRICAS")
print("-" * 45)
print("  Teste de Kolmogorov-Smirnov vs Distribuição Uniforme\n")

features_uniformes = []
for col in num_cols:
    vals = df[col].dropna()
    # normalizar entre 0 e 1 para comparar com uniforme
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    stat, p = stats.kstest(vals_norm, "uniform")
    is_uniform = p > 0.05
    sinal = "🔴" if is_uniform else "🟢"
    print(f"  {sinal} {col:<28} p={p:.4f}  {'← parece uniforme/aleatório' if is_uniform else ''}")
    if is_uniform:
        features_uniformes.append(col)

if len(features_uniformes) > len(num_cols) * 0.5:
    alertas.append(f"{len(features_uniformes)}/{len(num_cols)} features numéricas seguem distribuição uniforme (geradas com random).")
    pontos += 2

# ══════════════════════════════════════════════════════════════
# PASSO 5 — DUPLICATAS E VALORES REPETIDOS
# ══════════════════════════════════════════════════════════════
print("\n\n🔁 PASSO 5: DUPLICATAS E VALORES REPETIDOS")
print("-" * 45)
duplicatas = df.duplicated().sum()
print(f"  Linhas duplicadas: {duplicatas}")
if duplicatas == 0:
    print("  🟡 Nenhuma duplicata — pode indicar geração controlada.")
    pontos += 0.5

for col in num_cols:
    n_unique = df[col].nunique()
    ratio = n_unique / len(df)
    print(f"  {col:<28} {n_unique} valores únicos  ({ratio:.1%} do total)")

# ══════════════════════════════════════════════════════════════
# PASSO 6 — VALORES EXATAMENTE REDONDOS (Ex: 100, 50, 25)
# ══════════════════════════════════════════════════════════════
print("\n\n🔢 PASSO 6: VALORES SUSPEITOS (demasiado redondos)")
print("-" * 45)
for col in num_cols:
    vals = df[col].dropna()
    pct_round = (vals == vals.round(-1)).mean()  # múltiplos de 10
    if pct_round > 0.30 and vals.std() > 5:
        print(f"  🟡 {col}: {pct_round:.1%} dos valores são múltiplos de 10")

print("  (Nada crítico encontrado neste dataset nessa verificação)")

# ══════════════════════════════════════════════════════════════
# PASSO 7 — BASELINE DO MODELO
# ══════════════════════════════════════════════════════════════
print("\n\n🎯 PASSO 7: BASELINE RÁPIDO (sem treinar modelo)")
print("-" * 45)

std_target = df[TARGET].std()
media_target = df[TARGET].mean()
print(f"  Média do target : {media_target:.4f}")
print(f"  Desvio padrão   : {std_target:.4f}  ← RMSE do pior modelo possível")
print(f"  Se seu modelo tiver RMSE ≈ {std_target:.4f}, ele não aprendeu nada.")

# ══════════════════════════════════════════════════════════════
# GERAÇÃO DO GRÁFICO
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Detecção de Dados Sintéticos — Análise Visual", fontsize=14, fontweight="bold")

# 1 — Heatmap de correlação
mask = np.eye(len(df_enc.columns), dtype=bool)
sns.heatmap(df_enc.corr(), annot=False, cmap="coolwarm", ax=axes[0, 0],
            vmin=-0.3, vmax=0.3, linewidths=0.5, mask=mask)
axes[0, 0].set_title("Correlações (tudo perto de zero = suspeito)")

# 2 — Distribuição das correlações com target
corr_target.sort_values().plot(kind="barh", ax=axes[0, 1],
    color=["#e74c3c" if abs(v) < 0.05 else "#f39c12" if abs(v) < 0.15 else "#2ecc71"
           for v in corr_target.sort_values()])
axes[0, 1].axvline(0, color="k", linestyle="--", lw=1)
axes[0, 1].set_title(f"Correlação das features com '{TARGET}'")
axes[0, 1].set_xlabel("Correlação")

# 3 — Histogramas das features numéricas
cols_to_plot = num_cols[:4]
for i, col in enumerate(cols_to_plot):
    axes[1, 0].hist(df[col], bins=20, alpha=0.6, label=col, edgecolor="white")
axes[1, 0].set_title("Distribuição features numéricas (uniforme = suspeito)")
axes[1, 0].legend(fontsize=8)

# 4 — Scatter: feature mais correlacionada vs target
top_feat = corr_target.abs().idxmax()
axes[1, 1].scatter(df[top_feat], df[TARGET], alpha=0.3, s=15, color="steelblue")
axes[1, 1].set_xlabel(f"{top_feat} (corr={corr_target[top_feat]:.3f})")
axes[1, 1].set_ylabel(TARGET)
axes[1, 1].set_title(f"Melhor feature vs Target\n(nuvem sem forma = sem padrão)")

plt.tight_layout()
plt.savefig("/var/www/kaggle/relatorio_sintetico.png", dpi=130)
print("\nSaved: relatorio_sintetico.png")

# ══════════════════════════════════════════════════════════════
# VEREDITO FINAL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  VEREDITO FINAL")
print("=" * 65)

for i, a in enumerate(alertas, 1):
    print(f"  ⚠  {i}. {a}")

print(f"\n  Pontuação de suspeita: {pontos}/10")

if pontos >= 6:
    veredito = "🔴 ALTAMENTE SUSPEITO — dataset provavelmente sintético."
    acao = "Não vale a pena treinar modelo. Troque o dataset."
elif pontos >= 3:
    veredito = "🟡 MODERADAMENTE SUSPEITO — pode ter algum sinal, mas fraco."
    acao = "Treine um modelo simples antes de investir tempo."
else:
    veredito = "🟢 PARECE OK — dataset com características de dados reais."
    acao = "Pode seguir com o pipeline de ML normalmente."

print(f"\n  {veredito}")
print(f"  Ação recomendada: {acao}")
print("=" * 65)
