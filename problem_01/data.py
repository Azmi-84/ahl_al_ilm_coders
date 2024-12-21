import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from datasets import Dataset

def read_and_parse(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '|' in line:
                source, target = line.split('|', 1)
                source = source.strip('[bn]').strip()
                target = target.strip('[rm]').strip()
                data.append((source, target))
    return pd.DataFrame(data, columns=["source", "target"])

def normalize_text(text):
    return text.lower().strip()

file_path = 'data.csv'
df = read_and_parse(file_path)
df['source'] = df['source'].apply(normalize_text)
df['target'] = df['target'].apply(normalize_text)

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indic-bert")

def tokenize_and_encode(text, max_length=512):
    return tokenizer.encode(text, padding='max_length', truncation=True, max_length=max_length)

df['source_ids'] = df['source'].apply(lambda x: tokenize_and_encode(x))
df['target_ids'] = df['target'].apply(lambda x: tokenize_and_encode(x))

df = df[df['source'].str.len() > 0]
df = df[df['target'].str.len() > 0]

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print(train_dataset[0])
