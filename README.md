# AT&T Spam Detector

Automatic SMS spam detection using deep learning — from simple baseline to Transfer Learning.

---

## Context

AT&T users are constantly exposed to spam messages. This project builds an automated spam detector that classifies SMS as **spam** or **ham** (legitimate) based solely on message content.

## Dataset

**SMS Spam Collection** — `spam.csv`
- 5 572 SMS messages labeled as spam or ham
- Imbalanced: 87% ham / 13% spam
- Source: UCI Machine Learning Repository

## Models

Four models are built in order of increasing complexity:

| # | Model | F1 (spam) | Precision | Recall |
|---|-------|-----------|-----------|--------|
| 1 | Logistic Regression + TF-IDF | 0.942 | 0.958 | 0.926 |
| 2 | MLP (PyTorch) + TF-IDF | 0.926 | 0.926 | 0.926 |
| 3 | CNN + Embeddings (PyTorch) | 0.929 | 0.932 | 0.926 |
| 4 | DistilBERT — Transfer Learning | **0.965** | **1.000** | 0.933 |

## Key Results

- **Best model: DistilBERT** with F1 = 0.965 and a perfect Precision of 1.000 (zero false positives)
- **Logistic Regression outperforms MLP and CNN** — a reminder that complexity does not always mean better performance on structured text tasks
- **Main metric: F1-score** — accuracy alone is misleading on an imbalanced dataset (a model always predicting "ham" would reach 87% accuracy but 0% F1)

## How to Run

**Recommended: Google Colab (free GPU)**

1. Upload `spam.csv` to Google Drive
2. Open `spam_detector_ATT.ipynb` in Colab
3. Enable GPU: `Runtime → Change runtime type → T4 GPU`
4. Run all cells

**Local (CPU only — slow for DistilBERT)**

```bash
conda env create -f environment.yml
conda activate ml
jupyter lab
```

## 📁 Project Structure

```
├── spam.csv                  # Dataset
├── spam_detector_ATT.ipynb   # Main notebook
├── environment.yml           # Conda environment
└── README.md
```
