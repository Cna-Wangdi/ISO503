import os
import re
import pickle
import random
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers, optimizers

# 0. CONFIG

# Folder created after extracting domain_sentiment_data.tar.gz
BASE_DIR = "domain_sentiment_data/sorted_data_acl"

# Domains you want to include (these match the subfolders inside sorted_data_acl/)
DOMAINS = [
    "books",
    "dvd",
    "electronics",
    "kitchen_&_housewares",
]

MAX_VOCAB_SIZE = 20000   # keep top 20k most frequent words
MAX_SEQ_LEN = 200        # pad / truncate all reviews to 200 tokens
MIN_CHARS = 20           # drop reviews shorter than this (reduce noise)
TEST_SIZE = 0.15         # 15% of data as test set
VAL_SIZE = 0.15          # 15% of total as validation set (calculated from remaining)
RANDOM_SEED = 42

EMBED_DIM = 128
LSTM_UNITS = 256
BATCH_SIZE = 64
EPOCHS = 5              # can increase further if needed


# 1. LOAD DATA FROM SENTIMENT FILES

def load_reviews_from_sentiment_files(base_dir: str, domains: List[str]) -> List[Tuple[str, int]]:
    """
    Loads positive and negative reviews from the domain_sentiment_data structure.
    Each domain folder has:
        - positive.review
        - negative.review

    Returns:
        List of (text, label) tuples
            label = 1 for positive
            label = 0 for negative
    """
    examples: List[Tuple[str, int]] = []

    for domain in domains:
        domain_path = os.path.join(base_dir, domain)

        pos_path = os.path.join(domain_path, "positive.review")
        neg_path = os.path.join(domain_path, "negative.review")

        # Positive = 1
        if os.path.exists(pos_path):
            with open(pos_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        examples.append((text, 1))

        # Negative = 0
        if os.path.exists(neg_path):
            with open(neg_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        examples.append((text, 0))

    return examples


# 2. CLEANING FUNCTIONS

def basic_clean(text: str) -> str:
    text = text.lower()
    # keep letters, digits, spaces, and apostrophes
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noisy(text: str) -> bool:
    """
    Rules for noisy/too-short reviews.
    """
    if len(text) < MIN_CHARS:
        return True
    return False


# 3. MAIN PIPELINE

def main():
    # Set seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --------- Step 1: Load & combine & shuffle ---------
    print("Loading reviews from domain_sentiment_data...")
    all_reviews = load_reviews_from_sentiment_files(BASE_DIR, DOMAINS)

    print(f"Total reviews loaded before cleaning: {len(all_reviews)}")

    # Count original class balance
    orig_pos = sum(1 for _, y in all_reviews if y == 1)
    orig_neg = sum(1 for _, y in all_reviews if y == 0)
    print(f"Original class balance -> Positive: {orig_pos}, Negative: {orig_neg}")

    random.shuffle(all_reviews)

    # --------- Step 2: Clean + filter noisy reviews ---------
    cleaned_texts: List[str] = []
    labels: List[int] = []

    for text, label in all_reviews:
        cleaned = basic_clean(text)
        if not is_noisy(cleaned):
            cleaned_texts.append(cleaned)
            labels.append(label)

    print(f"Total reviews after cleaning/filtering: {len(cleaned_texts)}")

    # Check balance after cleaning
    pos_count = sum(1 for y in labels if y == 1)
    neg_count = sum(1 for y in labels if y == 0)
    print(f"After cleaning -> Positive: {pos_count}, Negative: {neg_count}")

    # --------- Step 3: Encode text + labels, pad sequences ---------
    print("Fitting tokenizer on cleaned text...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<UNK>")
    tokenizer.fit_on_texts(cleaned_texts)

    print("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(cleaned_texts)

    if len(sequences) == 0:
        raise ValueError("No sequences created. Check your dataset paths and cleaning rules.")

    print("Example cleaned text:", cleaned_texts[0][:200])
    print("Example token sequence:", sequences[0][:20])

    print("Padding/truncating sequences...")
    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post"
    )

    labels_array = np.array(labels)

    # --------- Step 4: Split into train / val / test ---------
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        padded_sequences,
        labels_array,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels_array
    )

    # Now split remaining into train & val
    val_ratio_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_of_temp,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Val size:   {X_val.shape[0]}")
    print(f"Test size:  {X_test.shape[0]}")

    # --------- Compute class weights (important if classes are imbalanced) ---------
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_array),
        y=labels_array
    )
    class_weights = {i: class_weights_arr[i] for i in range(len(class_weights_arr))}
    print("Class weights used:", class_weights)

    # --------- Step 5: Build the LSTM model ---------
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    print(f"Using vocab size: {vocab_size}")

    model = models.Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=EMBED_DIM,
            input_length=MAX_SEQ_LEN
        ),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=False)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")  # binary classification: pos/neg
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )

    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights   # <-- important
    )

    # --------- Step 7: Evaluate on test set ---------
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")
    print("\nTarget guide:")
    print("  ≥ 80%  = Pass level")
    print("  ≥ 90%  = Distinction level")
    print("  ~100%  = HD target (on your chosen test set)")

    # --------- Inference on new sentences ---------

    def predict_sentiment(sentence: str) -> str:
        cleaned = basic_clean(sentence)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
        prob = float(model.predict(pad, verbose=0)[0][0])
        label = "Positive" if prob >= 0.5 else "Negative"
        return f"{label} (prob={prob:.3f})"

    print("\nTesting model on some custom sentences:")
    examples = [
        "I absolutely love this product, it exceeded my expectations!",
        "This was a complete waste of money, very disappointed.",
        "It is okay, not great but not terrible either.",
        "The quality is horrible and I will never buy this again.",
        "Amazing value for the price, highly recommend!"
    ]

    for s in examples:
        print(f"\nText: {s}")
        print("Prediction:", predict_sentiment(s))

    # --------- Save model (for web app later) ---------
    model.save("sentiment_lstm_model.keras")

    with open("tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)

    print("\nModel saved as 'sentiment_lstm_model.keras' and tokenizer as 'tokenizer.pickle'.")


if __name__ == "__main__":
    main()