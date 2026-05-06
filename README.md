# 📰 Fake News Detection: Transformer-Based Contextual Analysis

This repository contains the complete implementation and experimental analysis for the project:

**"Fake News Detection Using Transformer Models: A Comparative Study with Traditional Methods."**

The project explores the shift from frequency-based linguistic analysis to deep contextual understanding using state-of-the-art Natural Language Processing (NLP) techniques.

---

## 🎯 Project Objectives

- **Comparative Analysis:** Measure the performance gap between traditional Machine Learning (Logistic Regression) and Deep Learning (BERT).
- **Contextual Awareness:** Evaluate how Attention Mechanisms improve the detection of subtle misinformation that keywords alone might miss.
- **Reproducibility:** Provide an end-to-end pipeline that handles data ingestion, model fine-tuning, and automated visualization.

---

## 📊 Experimental Performance

The following metrics were obtained using the ISOT Fake News Dataset. While traditional models perform strongly on this dataset, BERT demonstrates superior robustness in semantic classification.

| Model Architecture     | Accuracy | F1-Score | Complexity | Primary Advantage                              |
|----------------------|----------|----------|------------|-----------------------------------------------|
| Logistic Regression  | 0.9666   | 0.9660   | Low        | High speed, word-frequency focused            |
| BERT (Transformer)   | 0.9800   | 0.9792   | High       | Deep contextual/semantic understanding        |

---

## 📈 Visual Analysis & Interpretation

### 1. Training Convergence (Loss Curve)

The Training Loss Progression graph confirms a stable fine-tuning process.

- **Observations:** The loss decreased exponentially from ≈ 0.68 to ≈ 0.03.
- **Inference:** This indicates that the pre-trained weights of the BERT model effectively adapted to the specific linguistic nuances of the fake news dataset within a single epoch, avoiding both underfitting and overfitting.

---

### 2. Error Analysis (Confusion Matrices)

The visual heatmaps provide a breakdown of classification errors:

- **Logistic Regression:** Encountered ≈ 3% error rate. These errors often stem from "adversarial" fake news that uses neutral, high-frequency "Real" keywords to mask misinformation.
  
- **BERT Transformer:** Achieved near-perfect classification in test batches. By utilizing Self-Attention, BERT analyzes full sentence structure and identifies inconsistencies even when vocabulary appears legitimate.

---

## 📂 Repository Structure
FakeNewsProject/
├── data/               # Input Datasets (True.csv, Fake.csv)
├── results/            # Visualizations & Saved Metrics
│   ├── cm_baseline.png # Confusion Matrix: Logistic Regression
│   ├── cm_bert.png     # Confusion Matrix: BERT (Visual Failure Analysis)
│   ├── loss_curve.png  # BERT Training Progression Graph
│   └── config.json     # Model Architectural DNA (JSON format)
├── main.py             # Integrated Training & Evaluation Pipeline
└── README.md           # Professional Documentation (This file)

---

## ⚙️ Technical Methodology

### 1. Data Representation & Preprocessing

- **Text Concatenation**: The model analyzes the Title combined with the Body Text to capture the maximum available context.
- **Tokenization**: Uses `BertTokenizerFast`. Unlike simple splitting, this uses **WordPiece** tokenization, which handles out-of-vocabulary words by breaking them into sub-units.
- **Truncation**: Sequences are capped at 128 tokens. Research suggests that for fake news, the most critical stylistic markers are located in the headline and the lead paragraph.

### 2. Model Architecture: BERT-Base

- **Bidirectional Context**: Unlike previous models that read text left-to-right, BERT reads the entire sequence at once.
- **Attention Mechanism**: The model calculates "Attention Scores" between every word in a sentence, allowing it to understand the relationship between a subject and a distant verb or adjective.

## 🛠️ Installation & Setup

### Prerequisites

Ensure you have Python 3.9+ installed. The environment requires approximately 4GB of RAM for the Transformer model.

### 1. Install Dependencies

```bash
pip install pandas scikit-learn torch transformers accelerate matplotlib seaborn
```

### 2. Prepare Data

Place your `True.csv` and `Fake.csv` files inside a folder named `data/` in the root directory.

### 3. Run the Pipeline

```bash
python main.py
```

## ⚖️ Ethics & Responsible Use

- **Transparency**: Automated detection should be viewed as a "decision support" tool for human moderators.
- **Bias Awareness**: The model may reflect biases inherent in the ISOT dataset (e.g., specific political eras or sources).
- **Interpretability**: While BERT is highly accurate, it acts as a "black box." The included confusion matrices are the first step in understanding its decision-making logic.

---

**Author:** Vishnu Teja Gudipati  
**Riv ID:** A0000072432  
**Course:** COMP-690-AH2 - Topics in Computer Science  
**Date:** May 2026 improved performance over baseline models[cite: 14].
* [cite_start]Evaluate using F1-score and Accuracy[cite: 15].
