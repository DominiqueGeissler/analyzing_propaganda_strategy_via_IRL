"""
    Fine-tuning of BERT for stance detection
"""

import glob
import random
import re
import itertools

import numpy as np
import nums_from_string
import pandas as pd
import emoji

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TextClassificationPipeline

from datasets import Dataset
from datasets import load_metric

random.seed(42)

# Define paths
path_odds_ratio = "../data/odds_ratio/"
path_features_stance = "../data/stance_data/"

"""
  Extend vocabulary as in Kawintiranon & Singh (2021) https://aclanthology.org/2021.naacl-main.376.pdf
"""

# Select stance tokens
# Load and concatenate txt files with top words from log_odd_ratio analysis
files_top_words = glob.glob(path_odds_ratio + "top_words*")
top_words = list(itertools.chain.from_iterable([open(file, "r").readlines() for file in files_top_words]))

# Remove duplicates
top_words = list(set(top_words))
print(len(top_words))

# Save stance tokens
with open(str(path_odds_ratio + "stance_tokens.txt"), "w", encoding="utf-8") as f:
    for word in top_words:
        f.write(word)

stance_tokens = [re.sub(r"\n", "", token) for token in top_words]
stance_tokens.extend(["@user", "httpurl"])

# Extend vocabulary
# Load pre-trained BERTweet model
card = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(card, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(card)

# Current vocab size
print("Current vocab size " + str(len(tokenizer)))

tokenizer.add_tokens(stance_tokens)
model.resize_token_embeddings(len(tokenizer))

# New vocab size
print("New vocab size " + str(len(tokenizer)))

tokenizer.save_pretrained("bert_stance")
model.save_pretrained("bert_stance")

print("Vocabulary extended and saved")

"""
    Fine-tune bert_stance for stance detection
"""

# Load labeled data
df = pd.read_csv(path_features_stance + "data_stance_finetuning.csv", usecols=["labels", "text"])

# Combine Oppose (2) and Neither (3) labels for classification
df.loc[df["labels"] == 2, ["labels"]] = 0
df.loc[df["labels"] == 3, ["labels"]] = 0
df["labels"] = df["labels"].astype("int64")

# Create train test split
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.3)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True).remove_columns("text")

# Load model for Classification
card = "bert_stance"
model = AutoModelForSequenceClassification.from_pretrained(card, num_labels=2)

# Load evaluation metric ROC_AUC
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Define training arguments
training_args = TrainingArguments(
    output_dir="stance_detection",
    #overwrite_output_dir=True,
    evaluation_strategy="epoch",
    #save_strategy="epoch",
    #dataloader_num_workers=7,
    #load_best_model_at_end=True,
    #no_cuda=True,
    #num_train_epochs=2,
    #learning_rate=0.0005,
    #weight_decay=0.01,
    #logging_dir='\\log',
    #logging_steps=42
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Fine-tune model
trainer.train()

# Save fine-tuned model
trainer.save_model("bert_stance_finetuned")
print("Model finetuned and saved")

"""
    Predict stance of user data
"""


def create_stance_features(df):
    # Predict stance on new samples
    pred = pipe(list(df["body"]))

    # Reformat predictions
    scores = pd.concat([pd.DataFrame([x[1] for x in pred]),df]).drop(columns=["label", "body"]).rename(columns={"score": "stance_propaganda"})

    # Save predicted stance
    scores.to_pickle(path_features_stance + "raw_predicted_stance.gz")

    # Aggregate by user and average
    df_agg = scores.groupby("author_id").mean()

    # Save features by user file
    df_agg.to_pickle(path_features_stance + "by_user_predicted_stance.gz")

    print("Processed file.")

tweets_to_classify = pd.read_pickle(path_features_stance + "tweets_to_classify.gz")
# Prepare prediction pipline
card = "bert_stance_finetuned"
model = AutoModelForSequenceClassification.from_pretrained(card, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert_stance")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework="pt", return_all_scores=True)

# Compute predictions
create_stance_features(tweets_to_classify)
