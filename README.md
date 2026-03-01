# vit-emot-poc

PoC (Prova de Conceito) para classificação de emoções a partir de landmarks faciais
do dataset RAVDESS, usando modelos temporais (MLP, CNN1D, Transformer) e XAI
(Attention Rollout + Deletion Test). Projetado para execução em **CPU**.

## Estrutura do Projeto

```
vit-emot-poc/
├── data/
│   └── ravdess_landmarks_kaggle/
│       ├── 00_raw_kaggle_csv/      ← CSVs brutos do Kaggle (NÃO versionados)
│       ├── 01_processed_T100/      ← Dataset normalizado T=100 (.npz)
│       ├── 02_splits/              ← Split train/test (JSON)
│       └── 03_qc/                  ← Manifest e relatório de QC
├── notebooks/
│   ├── 01_ingest_qc_manifest.ipynb
│   ├── 02_preprocess_T100_dataset.ipynb
│   ├── 03_split_actor_holdout.ipynb
│   ├── 04_train_eval_models.ipynb
│   └── 05_xai_attention_deletion.ipynb
├── src/
│   ├── __init__.py
│   ├── ravdess_utils.py            ← Parsing RAVDESS, leitura CSV, manifest
│   ├── temporal.py                 ← Normalização temporal (T=100)
│   ├── metrics_utils.py            ← Métricas, seed, bootstrap
│   └── models.py                   ← MLP, CNN1D, EmoTransformer
├── reports/
│   ├── tables/                     ← Tabelas de resultados (.csv)
│   └── figures/                    ← Figuras geradas (.png)
├── runs/
│   └── poc_v1/
│       ├── metrics/                ← Métricas de treino e XAI
│       └── checkpoints/            ← Checkpoints dos modelos (.pt)
├── .gitignore
├── requirements.txt
└── README.md
```

## Dados

### Onde colocar os CSVs

Baixe os CSVs de facial landmark tracking do RAVDESS no Kaggle e coloque em:

```
data/ravdess_landmarks_kaggle/00_raw_kaggle_csv/
```

A estrutura interna pode conter subpastas (ex: `Actor_01/`, `Actor_02/`, etc.)
ou arquivos diretos — o notebook 01 faz descoberta recursiva.

Os dados **não são versionados** (estão no `.gitignore`).

## Setup

### 1. Criar ambiente virtual

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Registrar kernel Jupyter (opcional)

```bash
python -m ipykernel install --user --name vit-emot-poc
```

## Ordem de Execução

Execute os notebooks na ordem, a partir da pasta `notebooks/`:

| # | Notebook | Descrição | Output Principal |
|---|----------|-----------|------------------|
| 1 | `01_ingest_qc_manifest.ipynb` | Ingestão e QC dos CSVs | `03_qc/manifest.csv`, `03_qc/qc_report.csv` |
| 2 | `02_preprocess_T100_dataset.ipynb` | Normalização temporal T=100 | `01_processed_T100/dataset_T100.npz` |
| 3 | `03_split_actor_holdout.ipynb` | Split por ator (hold-out) | `02_splits/split_actor_holdout.json` |
| 4 | `04_train_eval_models.ipynb` | Treino e avaliação (3 modelos) | `runs/poc_v1/metrics/`, `reports/` |
| 5 | `05_xai_attention_deletion.ipynb` | XAI: atenção + deleção | `reports/figures/xai_*.png` |

## Modelos

| Modelo | Tipo | Parâmetros (aprox.) |
|--------|------|---------------------|
| FlatMLP | Baseline (achata T×D) | ~variável |
| TemporalCNN1D | CNN 1D temporal | ~variável |
| EmoTransformer | Transformer com CLS token | d_model=64, 2 layers, 4 heads |

## Métricas Geradas

- Accuracy, Balanced Accuracy, Macro F1
- Classification Report completo (precision/recall/F1 por classe)
- Confusion Matrix
- Bootstrap 95% CI para Macro F1 (200 reamostragens)
- Tempos de treino
- Métricas XAI (fidelidade por deleção)

## Outputs por Pasta

- **`data/.../03_qc/`**: manifest.csv, qc_report.csv, qc_distributions.png
- **`data/.../01_processed_T100/`**: dataset_T100.npz (X, y, actor_ids)
- **`data/.../02_splits/`**: split_actor_holdout.json
- **`runs/poc_v1/metrics/`**: metrics.csv, xai_fidelity_deletion.csv
- **`runs/poc_v1/checkpoints/`**: best_transformer.pt
- **`reports/tables/`**: results_table.csv
- **`reports/figures/`**: training_curves.png, confusion_matrix_*.png, f1_per_class.png, xai_*.png
