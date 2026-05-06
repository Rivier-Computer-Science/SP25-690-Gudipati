import pandas as pd
import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np

# --- MODEL DEFINITIONS ---

def get_bert_model(device):
    """Primary Model: BERT for sequence classification."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    return model.to(device)

# --- PIPELINE COMPONENTS ---

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # Optimization: 128 tokens is sufficient for fake news detection markers
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.labels = labels.tolist()
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generates a heatmap for visual failure analysis."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/{filename}')
    plt.close()

def run_pipeline():
    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. Data Ingestion
    print("[1/5] Loading and balancing data...")
    if not os.path.exists('data/True.csv') or not os.path.exists('data/Fake.csv'):
        print("ERROR: CSV files not found in 'data/' folder.")
        return

    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df['label'], fake_df['label'] = 1, 0
    
    # Use 10% sample to balance speed and accuracy
    df = pd.concat([true_df, fake_df]).sample(frac=0.1, random_state=42) 
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['title'] + " " + df['text'], df['label'], test_size=0.2, random_state=42
    )

    # 2. Logistic Regression Baseline
    print("[2/5] Training Baseline: Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)
    lr_preds = lr.predict(X_test_tfidf)
    print(f"      Baseline F1: {f1_score(y_test, lr_preds):.4f}")
    plot_confusion_matrix(y_test, lr_preds, "Logistic Regression", "cm_baseline.png")

    # 3. BERT Transformer Initialization
    print("[3/5] Initializing Transformer: BERT-Base...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = get_bert_model(device)
    
    # Small subsets for fast demonstration on CPU; increase for final production
    train_ds = NewsDataset(X_train[:300], y_train[:300], tokenizer) 
    test_ds = NewsDataset(X_test[:100], y_test[:100], tokenizer)

    # 4. Training BERT
    print("[4/5] Fine-tuning BERT (1 Epoch)...")
    try:
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        )
        
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        # 5. Visualizing Loss Curves & Final Evaluation
        print("[5/5] Generating Visualizations & Final Metrics...")
        history = trainer.state.log_history
        train_loss = [x['loss'] for x in history if 'loss' in x]
        
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss, label='Training Loss', color='orange')
        plt.title('BERT Training Loss Progression')
        plt.xlabel('Logging Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/loss_curve.png')
        plt.close()

        # Generate BERT Confusion Matrix for Failure Analysis
        raw_preds = trainer.predict(test_ds)
        bert_preds = np.argmax(raw_preds.predictions, axis=-1)
        plot_confusion_matrix(y_test[:100], bert_preds, "BERT Transformer", "cm_bert.png")

        print(f"\n--- EXECUTION SUCCESS ---")
        print(f"BERT Accuracy: {raw_preds.metrics['test_accuracy']:.4f}")
        print(f"BERT F1-Score: {raw_preds.metrics['test_f1']:.4f}")
        print("\nAll plots and metrics have been saved in the 'results/' folder.")
        
    except ImportError:
        print("\nFATAL ERROR: 'accelerate' library is missing.")
        print("Please run: pip install accelerate")

if __name__ == "__main__":
    run_pipeline()

if __name__ == "__main__":
    run_pipeline()
