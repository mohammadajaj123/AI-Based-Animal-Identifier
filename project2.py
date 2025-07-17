import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

DATASET_PATH = "dataset"
IMAGE_SIZE = (64, 64)
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_PCA = True
PCA_COMPONENTS = 0.95

def load_dataset():
    images, labels, class_names = [], [], []
    print("Loading dataset...")
    for class_name in sorted(os.listdir(DATASET_PATH)):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
            for file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMAGE_SIZE)
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    print(f" Error with {img_path}: {e}")
    print(f" Loaded {len(images)} images from {len(class_names)} classes.")
    return np.array(images), np.array(labels), class_names

def preprocess_images(images):
    processed = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        normalized = gray.astype('float32') / 255.0
        processed.append(normalized.flatten())
    return np.array(processed)

if __name__ == "__main__":
    X, y, class_names = load_dataset()

    if len(X) < 500 or len(class_names) < 3:
        print(" You need at least 500 images and 3+ classes.")
        exit()

    X_processed = preprocess_images(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_encoded
    )

    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f" PCA reduced dimensions to {X_train.shape[1]} features")

    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=10,
            min_samples_leaf=5, random_state=RANDOM_STATE),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=1000,
            solver='adam', alpha=0.001,
            learning_rate_init=0.001,
            early_stopping=True, random_state=RANDOM_STATE)
    }

    results = {}

    for name, model in models.items():
        print(f"\n Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        }

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} Confusion Matrix\nAccuracy: {acc:.4f}")
        plt.tight_layout()
        filename = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        plt.close()
        print(f" Saved confusion matrix: {filename}")

    results_df = pd.DataFrame(results).T
    print("\n Evaluation Matrix (All Models):")
    print(results_df.round(4))

    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(results.keys())
    metric_names = list(results[model_names[0]].keys())
    bar_width = 0.15
    index = np.arange(len(model_names))

    for i, metric in enumerate(metric_names):
        values = [results[model][metric] for model in model_names]
        ax.bar(index + i * bar_width, values, bar_width, label=metric)

    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title(" Model Performance Comparison")
    ax.set_xticks(index + bar_width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    plt.close()


    conf_matrix_nb = np.array([
        [131, 5, 24],
        [0, 145, 15],
        [0, 19, 141]
    ])

    accuracy = np.trace(conf_matrix_nb) / np.sum(conf_matrix_nb)

    precision_bikes  = 131 / (131 + 0 + 0)
    precision_planes = 145 / (5 + 145 + 19)
    precision_trains = 141 / (24 + 15 + 141)
    precision_macro = (precision_bikes + precision_planes + precision_trains) / 3

    recall_bikes  = 131 / (131 + 5 + 24)
    recall_planes = 145 / (0 + 145 + 15)
    recall_trains = 141 / (0 + 19 + 141)
    recall_macro = (recall_bikes + recall_planes + recall_trains) / 3

    f1_bikes  = 2 * (precision_bikes * recall_bikes) / (precision_bikes + recall_bikes)
    f1_planes = 2 * (precision_planes * recall_planes) / (precision_planes + recall_planes)
    f1_trains = 2 * (precision_trains * recall_trains) / (precision_trains + recall_trains)
    f1_macro = (f1_bikes + f1_planes + f1_trains) / 3

    print("\n Manual Evaluation from Image-based Confusion Matrix:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")

    print("\n All tasks complete.")
