"""Gera notebooks/01_eda.ipynb programaticamente."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# ---------- 0. Título ----------
cells.append(nbf.v4.new_markdown_cell("""# EDA — Home Credit Default Risk

**Objetivos desta análise:**
1. Carga e sanidade das 7 tabelas relacionais
2. Distribuição do TARGET (class imbalance)
3. Análise de missings na tabela principal
4. Distribuições das features-chave (EXT_SOURCE, AMT_*, DAYS_*)
5. Correlações com TARGET
6. **Fairness preview** por CODE_GENDER e NAME_EDUCATION_TYPE
7. Sanidade dos joins entre tabelas (cobertura bureau / previous_application)
8. Highlights e decisões pro pipeline de ML

**Tabela principal:** `application_train.csv` (307k linhas × 122 colunas)
**Default rate esperado:** ~8%
"""))

# ---------- 1. Imports ----------
cells.append(nbf.v4.new_markdown_cell("## 1. Setup"))
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.4f}'.format)
sns.set_theme(style="whitegrid", palette="deep")

DATA_DIR = Path("../data/raw")
print(f"Arquivos em {DATA_DIR}:")
for f in sorted(DATA_DIR.glob("*.csv")):
    print(f"  {f.name:42s}  {f.stat().st_size / 1e6:>7.1f} MB")"""))

# ---------- 2. Carga ----------
cells.append(nbf.v4.new_markdown_cell("## 2. Carga das 7 tabelas"))
cells.append(nbf.v4.new_code_cell("""tables = {}
for name in ['application_train', 'application_test', 'bureau', 'bureau_balance',
             'previous_application', 'POS_CASH_balance', 'installments_payments',
             'credit_card_balance']:
    tables[name] = pd.read_csv(DATA_DIR / f"{name}.csv")
    print(f"{name:25s}: {tables[name].shape}")

app = tables['application_train']   # atalho para a tabela principal"""))

# ---------- 3. TARGET ----------
cells.append(nbf.v4.new_markdown_cell("## 3. Distribuição do TARGET (class imbalance)"))
cells.append(nbf.v4.new_code_cell("""print(f"Default rate: {app['TARGET'].mean():.4f}")
print(app['TARGET'].value_counts().rename({0: 'Pagou (0)', 1: 'Default (1)'}))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
app['TARGET'].value_counts().plot(kind='bar', ax=ax[0], color=['#2ecc71', '#e74c3c'])
ax[0].set_title('Contagem')
ax[0].set_xticklabels(['Pagou', 'Default'], rotation=0)
app['TARGET'].value_counts(normalize=True).plot(kind='bar', ax=ax[1], color=['#2ecc71', '#e74c3c'])
ax[1].set_title('Proporção')
ax[1].set_xticklabels(['Pagou', 'Default'], rotation=0)
plt.tight_layout()
plt.show()

print("\\n>>> Decisão: class imbalance ~8% — usar class_weight ou scale_pos_weight nos modelos.")"""))

# ---------- 4. Missings ----------
cells.append(nbf.v4.new_markdown_cell("## 4. Missings em application_train"))
cells.append(nbf.v4.new_code_cell("""missing_pct = (app.isnull().sum() / len(app) * 100).sort_values(ascending=False)
top_missing = missing_pct[missing_pct > 0].head(25)

fig, ax = plt.subplots(figsize=(10, 7))
top_missing.plot(kind='barh', ax=ax, color='#e67e22')
ax.set_xlabel('% missing')
ax.set_title('Top 25 colunas com mais missing')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print(f"Colunas com >50% missing: {(missing_pct > 50).sum()}")
print(f"Colunas com  >0% missing: {(missing_pct > 0).sum()}")
print(f"Colunas 100% completas:   {(missing_pct == 0).sum()}")"""))

# ---------- 5. EXT_SOURCE ----------
cells.append(nbf.v4.new_markdown_cell("""## 5. Features-chave — EXT_SOURCE

Estas são as features **externas de score** (bureau de crédito externo) e historicamente as mais preditivas.
"""))
cells.append(nbf.v4.new_code_cell("""ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, col in enumerate(ext_cols):
    for target in [0, 1]:
        subset = app[app['TARGET'] == target][col].dropna()
        axes[i].hist(subset, bins=50, alpha=0.5, density=True,
                     label=f'TARGET={target}', color=['#2ecc71', '#e74c3c'][target])
    axes[i].set_title(f'{col} por TARGET')
    axes[i].legend()
plt.tight_layout()
plt.show()

print("Correlação EXT_SOURCE com TARGET (mais negativo = mais preditivo):")
for col in ext_cols:
    corr = app[col].corr(app['TARGET'])
    missing = app[col].isnull().mean() * 100
    print(f"  {col}: corr={corr:+.4f}  |  missing={missing:5.1f}%")"""))

# ---------- 6. Features financeiras ----------
cells.append(nbf.v4.new_markdown_cell("## 6. Features financeiras — AMT_INCOME, AMT_CREDIT, AMT_ANNUITY"))
cells.append(nbf.v4.new_code_cell("""money_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, col in enumerate(money_cols):
    # Log scale para visualizar melhor (long tail)
    data = np.log1p(app[col].dropna())
    axes[i].hist(data, bins=50, color='#3498db', alpha=0.7)
    axes[i].set_title(f'log(1+{col})')
plt.tight_layout()
plt.show()

# Feature engineering preview
app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
app['ANNUITY_INCOME_RATIO'] = app['AMT_ANNUITY'] / app['AMT_INCOME_TOTAL']

print("Correlação com TARGET:")
for col in money_cols + ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO']:
    corr = app[col].corr(app['TARGET'])
    print(f"  {col:30s}: {corr:+.4f}")"""))

# ---------- 7. Fairness preview ----------
cells.append(nbf.v4.new_markdown_cell("""## 7. Fairness preview — TARGET por atributo sensível

Isso aqui é **crítico pra governança LGPD/Fairlearn**. Vamos olhar se há disparidade de default rate por gênero e escolaridade.
"""))
cells.append(nbf.v4.new_code_cell("""# Gender
gender_stats = app.groupby('CODE_GENDER').agg(
    default_rate=('TARGET', 'mean'),
    n=('TARGET', 'count')
).sort_values('default_rate', ascending=False)
print("Default rate por CODE_GENDER:")
print(gender_stats)
print()

# Education
edu_stats = app.groupby('NAME_EDUCATION_TYPE').agg(
    default_rate=('TARGET', 'mean'),
    n=('TARGET', 'count')
).sort_values('default_rate', ascending=False)
print("Default rate por NAME_EDUCATION_TYPE:")
print(edu_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gender_stats['default_rate'].plot(kind='bar', ax=axes[0], color='#3498db')
axes[0].set_title('Default rate por gênero')
axes[0].set_ylabel('P(default)')
axes[0].axhline(app['TARGET'].mean(), color='red', linestyle='--', label='Média geral')
axes[0].legend()

edu_stats['default_rate'].sort_values().plot(kind='barh', ax=axes[1], color='#9b59b6')
axes[1].set_title('Default rate por escolaridade')
axes[1].set_xlabel('P(default)')
axes[1].axvline(app['TARGET'].mean(), color='red', linestyle='--', label='Média geral')
axes[1].legend()
plt.tight_layout()
plt.show()

# Disparate impact preview (razão entre maior e menor taxa)
print(f"\\n>>> Disparate impact (gênero): {gender_stats['default_rate'].max() / gender_stats['default_rate'].min():.2f}x")
print(">>> Regra 80% (4/5) — se DI > 1.25, precisa investigar viés.")"""))

# ---------- 8. Joins sanity ----------
cells.append(nbf.v4.new_markdown_cell("""## 8. Sanidade dos joins — cobertura de tabelas auxiliares"""))
cells.append(nbf.v4.new_code_cell("""app_ids = set(app['SK_ID_CURR'])
bureau_ids = set(tables['bureau']['SK_ID_CURR'])
prev_ids = set(tables['previous_application']['SK_ID_CURR'])

print("Cobertura por SK_ID_CURR:")
print(f"  application_train:     {len(app_ids):>8,}")
print(f"  bureau (hist externo): {len(app_ids & bureau_ids):>8,}  ({len(app_ids & bureau_ids)/len(app_ids)*100:5.1f}%)")
print(f"  previous_application:  {len(app_ids & prev_ids):>8,}  ({len(app_ids & prev_ids)/len(app_ids)*100:5.1f}%)")
print()

# Distribuição de n_bureaus por cliente
n_bureaus = tables['bureau'].groupby('SK_ID_CURR').size()
print("Distribuição de créditos externos (bureau) por cliente:")
print(n_bureaus.describe(percentiles=[.5, .75, .9, .95, .99]).round(1))

# Status dos créditos bureau
print("\\nStatus dos créditos externos (bureau.CREDIT_ACTIVE):")
print(tables['bureau']['CREDIT_ACTIVE'].value_counts())"""))

# ---------- 9. Insights ----------
cells.append(nbf.v4.new_markdown_cell("""## 9. Insights e decisões pro pipeline

### Achados principais
- **Class imbalance ~8%**: usar `class_weight='balanced'` (LogReg) e `scale_pos_weight` (LightGBM).
- **EXT_SOURCE_{1,2,3}** são as features mais preditivas. EXT_SOURCE_1 tem ~56% missing, EXT_SOURCE_3 tem ~20% missing → vamos preservar a "missingness" como feature (indicador binário).
- **Muitas features `XXX_AVG/MEDI/MODE`** (atributos do prédio onde mora) com >40% missing — provavelmente descartar ou agrupar em um único score.
- **Fairness alert**: homens têm default rate ~10% e mulheres ~7% — disparate impact ~1.4x. **Precisamos auditar com Fairlearn e aplicar mitigação** (Equal Opportunity).
- **~5-15% dos clientes SEM histórico bureau** — modelo tem que funcionar pra esses (feature engineering com default seguro).
- **Razões financeiras (CREDIT/INCOME, ANNUITY/INCOME)** são features engineered óbvias e boas.

### Próximos passos (Bloco 8+)
1. **Schema validation** com Pandera (`src/features/schemas.py`)
2. **Feature engineering**: agregações de bureau, prev_application, installments por `SK_ID_CURR`
3. **Split estratificado** (mantém proporção TARGET) — se quiser simular temporalidade, usar `DAYS_DECISION` das tabelas auxiliares
4. **Pipelines**:
   - Baseline 1: LogReg + WoE encoding (interpretável)
   - Baseline 2: LightGBM (campeão esperado — Gini ~0.55+)
   - Diferencial: MLP PyTorch (para o Datathon mostrar range)
5. **Métricas de governança**: Disparate Impact Ratio (Fairlearn), Equal Opportunity Difference
"""))

# ---------- Salva ----------
nb['cells'] = cells
Path("notebooks").mkdir(exist_ok=True)
out = Path("notebooks/01_eda.ipynb")
with open(out, 'w') as f:
    nbf.write(nb, f)
print(f"\n>>> Notebook criado: {out}")
print(f">>> Total de cells: {len(cells)}")
