

"""
Comparative Analysis of Audio Classification With MFCC and STFT Features Using Machine Learning Techniques
Hybrid Approach:

Current - UrbanSound8K + ECS-50 --> to GTZAN
RETEST CNN + add GAT
"""

import os
import numpy as np
import pandas as pd
import librosa
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, classification_report)

warnings.filterwarnings('ignore')
np.random.seed(42)

SAMPLE_RATE = 22050
N_MFCC = 40

# UrbanSound8K class labels
URBANSOUND_CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

def extract_mfcc_features(file_path, max_duration=30):
    """Extract MFCC features: 40 means + 40 stds = 80 features"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=max_duration)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=N_MFCC)

        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        return None

def extract_stft_features(file_path, max_duration=30):
    """Extract STFT features: mean + std of STFT magnitude"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=max_duration)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        stft = np.abs(librosa.stft(y_trimmed, n_fft=2048, hop_length=512))

        stft_mean = np.mean(stft, axis=1)
        stft_std = np.std(stft, axis=1)

        return np.concatenate([stft_mean, stft_std])
    except Exception as e:
        return None

def load_urbansound8k(data_path, feature_type='mfcc'):
    """
    Load UrbanSound8K dataset with specified features
    Filename format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav
    classID is 0-9 (second number in filename)
    """
    print(f"\nLoading UrbanSound8K with {feature_type.upper()} features...")

    extract_func = extract_mfcc_features if feature_type == 'mfcc' else extract_stft_features

    X = []
    y = []

    # Walk through all fold directories
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)

                # Parse filename: [fsID]-[classID]-[occurrenceID]-[sliceID].wav
                parts = filename.split('-')
                if len(parts) >= 2:
                    try:
                        class_id = int(parts[1])  # Second number is the class

                        if 0 <= class_id <= 9:  # Valid class IDs
                            features = extract_func(file_path)

                            if features is not None:
                                X.append(features)
                                y.append(class_id)
                    except ValueError:
                        continue

    print(f"Loaded {len(X)} samples from {len(URBANSOUND_CLASSES)} classes")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"  {URBANSOUND_CLASSES[class_id]}: {count} samples")

    return np.array(X), np.array(y), URBANSOUND_CLASSES

def load_esc50(data_path, feature_type='mfcc'):
    """Load ESC-50 dataset with specified features"""
    print(f"\nLoading ESC-50 with {feature_type.upper()} features...")

    extract_func = extract_mfcc_features if feature_type == 'mfcc' else extract_stft_features

    # Find metadata file
    meta_path = None
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file == 'esc50.csv' or file == 'meta.csv':
                meta_path = os.path.join(root, file)
                break
        if meta_path:
            break

    if not meta_path:
        print("ERROR: Could not find esc50.csv metadata file")
        return None, None, None

    print(f"Found metadata: {meta_path}")
    metadata = pd.read_csv(meta_path)

    # Find audio directory
    audio_dir = None
    for root, dirs, files in os.walk(data_path):
        if 'audio' in dirs:
            audio_dir = os.path.join(root, 'audio')
            break

    if not audio_dir:
        # Search for wav files
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_dir = root
                    break
            if audio_dir:
                break

    print(f"Audio directory: {audio_dir}")

    X = []
    y = []

    for idx, row in metadata.iterrows():
        filename = row['filename']
        file_path = os.path.join(audio_dir, filename)

        if not os.path.exists(file_path):
            # Try searching
            for root, dirs, files in os.walk(data_path):
                if filename in files:
                    file_path = os.path.join(root, filename)
                    break

        if os.path.exists(file_path):
            features = extract_func(file_path)

            if features is not None:
                X.append(features)
                y.append(row['target'])

    class_names = sorted(metadata['category'].unique().tolist())
    print(f"Loaded {len(X)} samples from {len(class_names)} classes")

    return np.array(X), np.array(y), class_names

def get_models():
    """Initialize all 7 models from the paper"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', C=10.0, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'ANN': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                            early_stopping=True, random_state=42, verbose=False)
    }

def evaluate_models(X, y, class_names, feature_type):
    """Train and evaluate all models"""
    if X is None or len(X) == 0:
        print("ERROR: No data to train on!")
        return pd.DataFrame()

    print(f"\nTraining models with {feature_type.upper()} features...")
    print(f"Data shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []
    models = get_models()

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Feature Type': feature_type.upper(),
            'Accuracy': acc,
            'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred)
        })

        print(f"  Accuracy: {acc:.4f}")

    return pd.DataFrame(results)

def main():
    print("="*70)
    print("Paper 3 Replication: Audio Classification")
    print("="*70)

    import kagglehub

    print("\nDownloading UrbanSound8K...")
    urbansound_path = kagglehub.dataset_download("chrisfilo/urbansound8k")

    print("\nDownloading ESC-50...")
    esc50_path = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")

    print(f"\nUrbanSound8K path: {urbansound_path}")
    print(f"ESC-50 path: {esc50_path}")

    # UrbanSound8K
    print("\n" + "="*70)
    print("DATASET 1: UrbanSound8K")
    print("="*70)

    X_us_mfcc, y_us_mfcc, classes_us = load_urbansound8k(urbansound_path, 'mfcc')

    if X_us_mfcc is not None:
        results_us_mfcc = evaluate_models(X_us_mfcc, y_us_mfcc, classes_us, 'MFCC')

        X_us_stft, y_us_stft, _ = load_urbansound8k(urbansound_path, 'stft')
        results_us_stft = evaluate_models(X_us_stft, y_us_stft, classes_us, 'STFT')
    else:
        results_us_mfcc = pd.DataFrame()
        results_us_stft = pd.DataFrame()

    # ESC-50
    print("\n" + "="*70)
    print("DATASET 2: ESC-50")
    print("="*70)

    X_esc_mfcc, y_esc_mfcc, classes_esc = load_esc50(esc50_path, 'mfcc')

    if X_esc_mfcc is not None:
        results_esc_mfcc = evaluate_models(X_esc_mfcc, y_esc_mfcc, classes_esc, 'MFCC')

        X_esc_stft, y_esc_stft, _ = load_esc50(esc50_path, 'stft')
        results_esc_stft = evaluate_models(X_esc_stft, y_esc_stft, classes_esc, 'STFT')
    else:
        results_esc_mfcc = pd.DataFrame()
        results_esc_stft = pd.DataFrame()

    # Combine results
    all_results = []

    if not results_us_mfcc.empty:
        all_results.append(results_us_mfcc.assign(Dataset='UrbanSound8K'))
        all_results.append(results_us_stft.assign(Dataset='UrbanSound8K'))

    if not results_esc_mfcc.empty:
        all_results.append(results_esc_mfcc.assign(Dataset='ESC-50'))
        all_results.append(results_esc_stft.assign(Dataset='ESC-50'))

    if all_results:
        all_results = pd.concat(all_results)

        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(all_results.to_string(index=False))

        all_results.to_csv('replication_results.csv', index=False)
        print("\nResults saved to replication_results.csv")

        print("\n" + "="*70)
        print("COMPARISON WITH PAPER")
        print("="*70)
        print("Paper reported (UrbanSound8K with MFCC):")
        print("  ANN: 91.41%")

        us_results = all_results[all_results['Dataset'] == 'UrbanSound8K']
        if not us_results.empty:
            best_us = us_results['Accuracy'].max()
            best_model = us_results.loc[us_results['Accuracy'].idxmax(), 'Model']
            print(f"\nOur best (UrbanSound8K):")
            print(f"  {best_model}: {best_us*100:.2f}%")
            print(f"  Difference: {abs(0.9141 - best_us)*100:.2f}%")

        return all_results
    else:
        print("\nERROR: No results generated")
        return None

if __name__ == '__main__':
    results = main()
