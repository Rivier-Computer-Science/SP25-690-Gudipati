import pandas as pd
import torch
import torch.nn as nn
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np

# --- MODEL DEFINITIONS ---

def get_bert_model(device):
    """Primary Model: BERT for sequence classification."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    return model.to(device)

class MLPModel(nn.Module):
    """Baseline Neural Network (MLP)."""
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
    def forward(self, x):
        return self.network(x)

# --- PIPELINE COMPONENTS ---

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # Optimization: Reduced max_length to 128 to speed up training significantly
        self.encodings = tokenizer(
            list(texts), 
            truncation=True, 
            padding=True, 
            max_length=max_len
        )
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

def run_pipeline():
    # Detect hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Check Data
    print("[1/5] Checking data files...")
    if not os.path.exists('data/True.csv') or not os.path.exists('data/Fake.csv'):
        print("ERROR: CSV files not found in 'data/' folder.")
        return

    # 2. Load Data
    print("[2/5] Loading and splitting data...")
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df['label'], fake_df['label'] = 1, 0
    
    # We sample 10% for the baseline, but can keep transformer subset tighter for speed
    df = pd.concat([true_df, fake_df]).sample(frac=0.1, random_state=42) 
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['title'] + " " + df['text'], df['label'], test_size=0.2, random_state=42
    )

    # 3. Logistic Regression Baseline
    print("[3/5] Running Logistic Regression baseline...")
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    lr_preds = lr.predict(X_test_tfidf)
    print(f"      Baseline Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print(f"      Baseline F1-Score: {f1_score(y_test, lr_preds):.4f}")

    # 4. BERT Transformer
    print("[4/5] Initializing BERT (Using Fast Tokenizer)...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = get_bert_model(device)
    
    # Speed Optimization: 
    # 1. Shortened sequence length (max_len=128)
    # 2. Focused sample sizes for transformer training
    train_ds = NewsDataset(X_train[:300], y_train[:300], tokenizer, max_len=128) 
    test_ds = NewsDataset(X_test[:100], y_test[:100], tokenizer, max_len=128)

    print("[5/5] Fine-tuning BERT...")
    try:
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=8, # Increased batch size for speed
            per_device_eval_batch_size=8,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        )
        
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        print("\n--- Final BERT Evaluation ---")
        results = trainer.evaluate()
        print(f"BERT Accuracy: {results['eval_accuracy']:.4f}")
        print(f"BERT F1-Score: {results['eval_f1']:.4f}")
        
    except ImportError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please run: pip install 'accelerate>=1.1.0'")
        return

    print("\n--- Project Execution Complete ---")

if __name__ == "__main__":
    run_pipeline()