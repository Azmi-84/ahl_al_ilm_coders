import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
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

tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def tokenize_and_encode(text, max_length=128):
    return tokenizer.encode(text, padding='max_length', truncation=True, max_length=max_length)

df['source_ids'] = df['source'].apply(lambda x: tokenize_and_encode(x))
df['target_ids'] = df['target'].apply(lambda x: tokenize_and_encode(x))

df = df[df['source'].str.len() > 0]
df = df[df['target'].str.len() > 0]

max_length = 128
df = df[df['source_ids'].apply(len) <= max_length]
df = df[df['target_ids'].apply(len) <= max_length]

def add_special_tokens(encoded_sequence):
    cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.pad_token_id
    return [cls_token_id] + encoded_sequence + [sep_token_id]

df['source_ids'] = df['source_ids'].apply(add_special_tokens)
df['target_ids'] = df['target_ids'].apply(add_special_tokens)

def create_attention_mask(ids, max_length=128):
    return [1 if id != tokenizer.pad_token_id else 0 for id in ids] + [0] * (max_length - len(ids))

df['source_attention_mask'] = df['source_ids'].apply(lambda x: create_attention_mask(x, max_length))
df['target_attention_mask'] = df['target_ids'].apply(lambda x: create_attention_mask(x, max_length))

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'input_ids': row['source_ids'],
            'attention_mask': row['source_attention_mask'],
            'labels': row['target_ids']
        }

train_dataset = TranslationDataset(train_df)
val_dataset = TranslationDataset(val_df)

train_df.to_csv("train_preprocessed.csv", index=False)
val_df.to_csv("val_preprocessed.csv", index=False)

for i in range(3):
    print(f"Source: {tokenizer.decode(train_dataset[i]['input_ids'])}")
    print(f"Target: {tokenizer.decode(train_dataset[i]['labels'])}")

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
    fp16=False,
    gradient_accumulation_steps=4
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()

results = trainer.evaluate()
print(results)

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
