* [cite_start]**Bold Text**: Wrap words in double asterisks (`**Bold**`) to highlight key metrics like **Accuracy** and **F1-score**[cite: 15, 45].

### 2. Standard Organized Template
You can copy and paste this structure directly into your `README.md` file in VS Code:

```markdown
# Fake News Detection Using Transformer Models

## 1. Project Overview
[cite_start]This project classifies news as real or fake using BERT and traditional baselines[cite: 4, 8].

## 2. Dataset
We use the ISOT Fake News Dataset containing 44,898 articles.
* **True Articles**: 21,417
* **Fake Articles**: 23,481

## 3. Implementation
[cite_start]The pipeline follows these stages[cite: 3]:
1. Data Ingestion    
2. [cite_start]Preprocessing (TF-IDF) [cite: 29]
3. [cite_start]Training (LR, MLP, BERT) [cite: 27, 35]
4. [cite_start]Evaluation [cite: 40]

## 4. Usage
```bash
pip install -r requirements.txt
python main.py
```

## 5. Success Criteria
* [cite_start]Achieve improved performance over baseline models[cite: 14].
* [cite_start]Evaluate using F1-score and Accuracy[cite: 15].
